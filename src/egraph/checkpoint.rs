//! Checkpoints: undoing every change made to an [`EGraph`] since a point in time.
//!
//! While a checkpoint is open the e-graph keeps a *trail*, a log of undo records, one
//! per change, each restoring exactly the state before its change. A rollback undoes
//! the records in reverse and then discards what was created since the checkpoint
//! (ids, e-nodes, e-classes). No trail is kept while no checkpoint is open.
//!
//! A record is proportional to the change it undoes, never to the size of a class:
//! a union records the lengths of the lists it concatenates, and a rebuild records
//! the nodes it canonicalized, moved or deduplicated with their positions. Undoing
//! a record costs at most what the change cost.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// The trail of an e-graph, `None` while no checkpoint is open. A clone of an
/// e-graph has no open checkpoints: checkpoints belong to the e-graph they were
/// opened on, so the clone of a trail is empty.
pub(crate) struct Recording<L: Language, D>(Option<Box<Trail<L, D>>>);

impl<L: Language, D> Default for Recording<L, D> {
    fn default() -> Self {
        Recording(None)
    }
}

impl<L: Language, D> Clone for Recording<L, D> {
    fn clone(&self) -> Self {
        Recording(None)
    }
}

impl<L: Language, D> std::ops::Deref for Recording<L, D> {
    type Target = Option<Box<Trail<L, D>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<L: Language, D> std::ops::DerefMut for Recording<L, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Every checkpoint open on an e-graph, and how to undo what happened since.
pub(crate) struct Trail<L: Language, D> {
    /// Captured by [`EGraph::checkpoint`], the only place that needs `D: Clone`.
    clone_data: fn(&D) -> D,
    frames: Vec<Frame>,
    log: Vec<Undo<L, D>>,
}

#[derive(Clone, Copy, Debug)]
struct Frame {
    /// Unique among all checkpoints of the process, so a token can only close the
    /// checkpoint it was issued for.
    id: u64,
    /// Number of ids (e-nodes) when the checkpoint was opened.
    size: usize,
    log_len: usize,
    uf_log_len: usize,
    /// The end of the change log, `None` if changes were not tracked.
    changes_end: Option<usize>,
}

/// Source of [`Frame::id`]s.
static NEXT_FRAME: AtomicU64 = AtomicU64::new(0);

/// One undo record.
#[derive(Clone)]
enum Undo<L: Language, D> {
    /// `memo[node]` was `old` (`None`: absent).
    Memo { node: L, old: Option<Id> },
    /// `classes_by_op[op]` gained (`listed`) or lost `id`.
    Listed {
        op: L::Discriminant,
        id: Id,
        listed: bool,
    },
    /// An e-node was pushed onto the parent list of `class`.
    ParentPush { class: Id },
    /// The analysis data of `class` was `old`.
    Data { class: Id, old: D },
    /// `loser` (with data `loser_data`) was merged into `root`. Each of the node and
    /// parent lists of the two was concatenated the way [`concat_vecs`] does it: the
    /// shorter appended to the longer, which then belongs to `root`.
    Union {
        root: Id,
        loser: Id,
        loser_data: D,
        nodes: Concat,
        parents: Concat,
    },
    /// A rebuild canonicalized, sorted and deduplicated the nodes of `class`.
    Rebuilt {
        class: Id,
        delta: Box<RebuildDelta<L>>,
    },
    /// The whole class before it was borrowed mutably.
    Class { class: Id, nodes: Vec<L>, data: D },
    /// The position of a change-log subscriber was `old` (`None`: it had none).
    Seen { subscriber: Symbol, old: Option<usize> },
}

/// How two lists were concatenated: `root` and `loser` long, and whether the
/// loser's list came first (it was the longer one).
#[derive(Clone, Copy)]
struct Concat {
    root: usize,
    loser: usize,
    loser_first: bool,
}

impl Concat {
    fn of<T>(root: &[T], loser: &[T]) -> Self {
        Concat {
            root: root.len(),
            loser: loser.len(),
            loser_first: root.len() < loser.len(),
        }
    }

    /// Splits the concatenation `list` back into the root's and the loser's lists.
    fn split<T>(self, list: &mut Vec<T>) -> Vec<T> {
        debug_assert_eq!(list.len(), self.root + self.loser);
        if self.loser_first {
            let root = list.split_off(self.loser);
            std::mem::replace(list, root)
        } else {
            list.split_off(self.root)
        }
    }
}

/// What a rebuild did to a node list `before`, to get it back from the result:
/// the nodes it canonicalized (index in `before` and the node there), the nodes
/// the sort moved (index before and after the sort), and the duplicates it dropped
/// (index in the sorted list and the node).
///
/// The nodes not moved keep their relative order, so the moved ones and their
/// positions are all a rollback needs. They are the canonicalized nodes and those
/// out of order: for the usual shape, a sorted run followed by a few appended
/// nodes, that is the appended nodes alone.
#[derive(Clone)]
pub(crate) struct RebuildDelta<L> {
    canonicalized: Vec<(usize, L)>,
    moved: Vec<(usize, usize)>,
    dropped: Vec<(usize, L)>,
}

impl<L: Language> RebuildDelta<L> {
    fn is_empty(&self) -> bool {
        self.canonicalized.is_empty() && self.moved.is_empty() && self.dropped.is_empty()
    }

    /// Canonicalizes, sorts and deduplicates `nodes` (as a rebuild does), returning
    /// the record of how.
    fn rebuild(nodes: &mut Vec<L>, uf: &mut UnionFind) -> Self {
        let mut delta = RebuildDelta {
            canonicalized: vec![],
            moved: vec![],
            dropped: vec![],
        };
        // canonicalize; a changed node, or one smaller than the last node kept in
        // place, moves
        let mut moves = Vec::with_capacity(nodes.len());
        let mut last_kept: Option<usize> = None;
        for i in 0..nodes.len() {
            let node = &mut nodes[i];
            let changed = node.any(|c| uf.find(c) != c);
            if changed {
                delta.canonicalized.push((i, node.clone()));
                node.update_children(|c| uf.find_mut(c));
            }
            let moved = changed || last_kept.is_some_and(|k| nodes[i] < nodes[k]);
            if !moved {
                last_kept = Some(i);
            }
            moves.push(moved);
        }
        if moves.iter().any(|&m| m) {
            let old = std::mem::take(nodes);
            let mut moved: Vec<(usize, L)> = vec![];
            let mut kept: Vec<L> = Vec::with_capacity(old.len());
            for (i, node) in old.into_iter().enumerate() {
                if moves[i] {
                    moved.push((i, node));
                } else {
                    kept.push(node);
                }
            }
            // stable: equal nodes keep their original order
            moved.sort_by(|a, b| a.1.cmp(&b.1));
            nodes.reserve(kept.len() + moved.len());
            let mut moved = moved.into_iter().peekable();
            for node in kept {
                while let Some((i, m)) = moved.next_if(|(_, m)| *m < node) {
                    delta.moved.push((i, nodes.len()));
                    nodes.push(m);
                }
                nodes.push(node);
            }
            for (i, m) in moved {
                delta.moved.push((i, nodes.len()));
                nodes.push(m);
            }
        }
        let mut i = 0;
        nodes.dedup_by(|a, b| {
            i += 1;
            let duplicate = a == b;
            if duplicate {
                delta.dropped.push((i, a.clone()));
            }
            duplicate
        });
        delta
    }

    /// Restores the node list a [`rebuild`](Self::rebuild) started from.
    fn undo(self, nodes: &mut Vec<L>) {
        // the duplicates, at their sorted positions (recorded in increasing order)
        if !self.dropped.is_empty() {
            let deduped = std::mem::take(nodes);
            nodes.reserve(deduped.len() + self.dropped.len());
            let mut dropped = self.dropped.into_iter().peekable();
            for node in deduped {
                nodes.push(node);
                while let Some((_, d)) = dropped.next_if(|(i, _)| *i == nodes.len()) {
                    nodes.push(d);
                }
            }
            debug_assert!(dropped.next().is_none());
        }
        // the moved nodes, back to their positions before the sort
        if !self.moved.is_empty() {
            let mut sorted: Vec<Option<L>> = std::mem::take(nodes).into_iter().map(Some).collect();
            let mut back: Vec<Option<L>> = Vec::with_capacity(sorted.len());
            back.resize_with(sorted.len(), || None);
            for &(before, after) in &self.moved {
                back[before] = sorted[after].take();
            }
            let mut kept = sorted.into_iter().flatten();
            nodes.extend(
                back.into_iter()
                    .map(|slot| slot.unwrap_or_else(|| kept.next().unwrap())),
            );
        }
        for (i, node) in self.canonicalized {
            nodes[i] = node;
        }
    }
}

impl<L: Language, D> Trail<L, D> {
    pub(crate) fn memo(&mut self, node: &L, old: Option<Id>) {
        self.log.push(Undo::Memo {
            node: node.clone(),
            old,
        });
    }

    pub(crate) fn listed(&mut self, op: L::Discriminant, id: Id, listed: bool) {
        self.log.push(Undo::Listed { op, id, listed });
    }

    pub(crate) fn parent_push(&mut self, class: Id) {
        self.log.push(Undo::ParentPush { class });
    }

    pub(crate) fn data(&mut self, class: Id, old: &D) {
        let old = (self.clone_data)(old);
        self.log.push(Undo::Data { class, old });
    }

    /// Records a union of `loser` into `root`, before the lists are concatenated.
    pub(crate) fn union(&mut self, root: &EClass<L, D>, loser: &EClass<L, D>) {
        self.data(root.id, &root.data);
        let loser_data = (self.clone_data)(&loser.data);
        self.log.push(Undo::Union {
            root: root.id,
            loser: loser.id,
            loser_data,
            nodes: Concat::of(&root.nodes, &loser.nodes),
            parents: Concat::of(&root.parents, &loser.parents),
        });
    }

    /// Canonicalizes, sorts and deduplicates the nodes of `class`, recording how.
    pub(crate) fn rebuild_class(&mut self, class: Id, nodes: &mut Vec<L>, uf: &mut UnionFind) {
        let delta = RebuildDelta::rebuild(nodes, uf);
        if !delta.is_empty() {
            self.log.push(Undo::Rebuilt {
                class,
                delta: Box::new(delta),
            });
        }
    }

    pub(crate) fn seen(&mut self, subscriber: Symbol, old: Option<usize>) {
        self.log.push(Undo::Seen { subscriber, old });
    }

    /// Records `class` whole before it is borrowed mutably.
    pub(crate) fn class(&mut self, class: &EClass<L, D>) {
        self.log.push(Undo::Class {
            class: class.id,
            nodes: class.nodes.clone(),
            data: (self.clone_data)(&class.data),
        });
    }
}

/// An open checkpoint of an [`EGraph`], returned by [`EGraph::checkpoint`] and
/// consumed by [`EGraph::rollback`] or [`EGraph::commit`].
#[must_use = "a checkpoint is closed by `rollback` or `commit`"]
#[derive(Debug)]
pub struct Checkpoint {
    depth: usize,
    id: u64,
}

impl<L: Language, N: Analysis<L>> EGraph<L, N> {
    /// Opens a checkpoint: [`rollback`](EGraph::rollback) restores the e-graph to
    /// its state now, [`commit`](EGraph::commit) keeps the changes made since.
    /// Checkpoints nest; they are closed in the reverse order of opening.
    ///
    /// The e-graph is [rebuilt](EGraph::rebuild) first if any work is pending. While
    /// a checkpoint is open every change also records how to undo it, at a cost
    /// proportional to the change (a mutable borrow of a class, through
    /// `egraph[id]` or [`classes_mut`](EGraph::classes_mut), records the whole
    /// class, as the borrower may change anything); with no checkpoint open nothing
    /// is recorded.
    /// State held by the [`Analysis`] value itself (rather than in the per-class
    /// data) is not restored.
    ///
    /// The returned token closes exactly this checkpoint of this e-graph; closing
    /// it any other way panics. A token that is dropped instead leaves the
    /// checkpoint open (and recording) until an enclosing one is closed. A clone of
    /// the e-graph has no open checkpoints.
    ///
    /// Explanations are not supported: panics if they are enabled.
    ///
    /// ```
    /// use egg::{*, SymbolLang as S};
    /// let mut egraph = EGraph::<S, ()>::default();
    /// let a = egraph.add(S::leaf("a"));
    /// let b = egraph.add(S::leaf("b"));
    /// let checkpoint = egraph.checkpoint();
    /// egraph.union(a, b);
    /// egraph.add(S::new("f", vec![a]));
    /// egraph.rebuild();
    /// assert_eq!(egraph.find(a), egraph.find(b));
    /// egraph.rollback(checkpoint);
    /// assert_ne!(egraph.find(a), egraph.find(b));
    /// assert_eq!(egraph.number_of_classes(), 2);
    /// ```
    pub fn checkpoint(&mut self) -> Checkpoint
    where
        N::Data: Clone,
    {
        assert!(
            self.explain.is_none(),
            "checkpoints are not supported with explanations enabled"
        );
        // Restore the invariants first if anything is pending (a mutable borrow or
        // `set_analysis_data` queues work without clearing `clean`): the rollback
        // restores this state and drops every queue.
        if !self.clean
            || !self.pending.is_empty()
            || !self.analysis_pending.is_empty()
            || !self.dirty.is_empty()
            || self.reindex_all
        {
            #[cfg(test)]
            tests::CHECKPOINT_REBUILDS.with(|n| n.set(n.get() + 1));
            self.rebuild();
        }
        let size = self.unionfind.size();
        let frame = Frame {
            id: NEXT_FRAME.fetch_add(1, Ordering::Relaxed),
            size,
            log_len: self.trail.as_ref().map_or(0, |t| t.log.len()),
            uf_log_len: self.unionfind.log_len(),
            changes_end: self.changes.as_ref().map(|log| log.end()),
        };
        if let (None, Some(log)) = (&*self.trail, &mut self.changes) {
            log.pinned = Some(log.lowest_needed());
        }
        let trail = self.trail.get_or_insert_with(|| {
            Box::new(Trail {
                clone_data: N::Data::clone,
                frames: Vec::new(),
                log: Vec::new(),
            })
        });
        trail.frames.push(frame);
        self.unionfind.log_writes_below(size);
        Checkpoint {
            depth: trail.frames.len() - 1,
            id: frame.id,
        }
    }

    /// The frame `checkpoint` was issued for. Panics if it is not open on this
    /// e-graph (closed by an enclosing rollback or commit, or another e-graph's).
    fn frame(&self, checkpoint: &Checkpoint) -> Frame {
        let frame = self
            .trail
            .as_ref()
            .and_then(|t| t.frames.get(checkpoint.depth))
            .filter(|f| f.id == checkpoint.id);
        *frame.expect("checkpoint is not open on this e-graph")
    }

    /// The number of open checkpoints.
    pub fn open_checkpoints(&self) -> usize {
        self.trail.as_ref().map_or(0, |t| t.frames.len())
    }

    /// Restores the e-graph to its state when `checkpoint` was opened, closing it
    /// and every checkpoint opened after it. Ids created since are no longer valid.
    /// The restored e-graph is clean. Costs time proportional to the changes undone.
    pub fn rollback(&mut self, checkpoint: Checkpoint) {
        let frame = self.frame(&checkpoint);
        let mut trail = self.trail.take().unwrap();

        for undo in trail.log.drain(frame.log_len..).rev() {
            match undo {
                Undo::Memo { node, old } => match old {
                    Some(id) => {
                        self.memo.insert(node, id);
                    }
                    None => {
                        self.memo.remove(&node);
                    }
                },
                Undo::Listed { op, id, listed } => {
                    let ids = self.classes_by_op.entry(op).or_default();
                    if listed {
                        ids.remove(&id);
                    } else {
                        ids.insert(id);
                    }
                }
                Undo::ParentPush { class } => {
                    self.classes.get_mut(&class).unwrap().parents.pop();
                }
                Undo::Data { class, old } => self.classes.get_mut(&class).unwrap().data = old,
                Undo::Union {
                    root,
                    loser,
                    loser_data,
                    nodes,
                    parents,
                } => {
                    let root = self.classes.get_mut(&root).unwrap();
                    let loser = EClass {
                        id: loser,
                        nodes: nodes.split(&mut root.nodes),
                        data: loser_data,
                        parents: parents.split(&mut root.parents),
                    };
                    self.classes.insert(loser.id, loser);
                }
                Undo::Rebuilt { class, delta } => {
                    delta.undo(&mut self.classes.get_mut(&class).unwrap().nodes);
                }
                Undo::Class { class, nodes, data } => {
                    let class = self.classes.get_mut(&class).unwrap();
                    class.nodes = nodes;
                    class.data = data;
                }
                Undo::Seen { subscriber, old } => {
                    self.changes.as_mut().unwrap().set_seen(subscriber, old);
                }
            }
        }

        let size = frame.size;
        self.unionfind.rollback(frame.uf_log_len, size);
        self.nodes.truncate(size);
        self.classes.truncate(size);
        match frame.changes_end {
            Some(end) => self.changes.as_mut().unwrap().truncate(end),
            None => self.changes = None,
        }
        self.pending.clear();
        while self.analysis_pending.pop().is_some() {}
        self.dirty.clear();
        self.unindexed.clear();
        self.reindex_all = false;
        self.clean = true;

        trail.frames.truncate(checkpoint.depth);
        self.reopen(trail);
    }

    /// Closes `checkpoint` (and every checkpoint opened after it), keeping the changes
    /// made since. An enclosing checkpoint still undoes them.
    pub fn commit(&mut self, checkpoint: Checkpoint) {
        self.frame(&checkpoint);
        let mut trail = self.trail.take().unwrap();
        trail.frames.truncate(checkpoint.depth);
        self.reopen(trail);
    }

    /// Keeps `trail` if a checkpoint is still open, and drops every record otherwise.
    fn reopen(&mut self, trail: Box<Trail<L, N::Data>>) {
        match trail.frames.last() {
            Some(frame) => {
                self.unionfind.log_writes_below(frame.size);
                *self.trail = Some(trail);
            }
            None => {
                self.unionfind.clear_log();
                if let Some(log) = &mut self.changes {
                    log.pinned = None;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    define_language! {
        enum Math {
            "+" = Add([Id; 2]),
            "f" = F([Id; 1]),
            Num(i32),
            Symbol(Symbol),
        }
    }

    /// Constant folding whose `modify` adds and unions, so rebuilds mutate the
    /// e-graph through the analysis too.
    #[derive(Clone, Default)]
    struct Fold;
    impl Analysis<Math> for Fold {
        type Data = Option<i32>;
        fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
            merge_max(to, from)
        }
        fn make(egraph: &mut EGraph<Math, Self>, enode: &Math, _id: Id) -> Self::Data {
            let x = |i: &Id| egraph[*i].data;
            match enode {
                Math::Num(n) => Some(*n),
                Math::Add([a, b]) => Some(x(a)?.wrapping_add(x(b)?) % 8),
                _ => None,
            }
        }
        fn modify(egraph: &mut EGraph<Math, Self>, id: Id) {
            if let Some(n) = egraph[id].data {
                let lit = egraph.add(Math::Num(n));
                egraph.union(id, lit);
            }
        }
    }

    type G = EGraph<Math, Fold>;

    /// Asserts `a` and `b` are the same e-graph, field by field, including the order
    /// `classes()` and `classes_for_op` iterate in, and that both have nothing
    /// pending.
    fn assert_identical(a: &G, b: &G) {
        assert_eq!(a.unionfind.parents(), b.unionfind.parents());
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.memo, b.memo);
        let classes = |g: &G| g.classes().map(|c| c.id).collect::<Vec<_>>();
        assert_eq!(classes(a), classes(b), "class order");
        for class in a.classes() {
            let other = &b.classes[&class.id];
            assert_eq!(class.nodes, other.nodes, "nodes of {}", class.id);
            assert_eq!(class.data, other.data, "data of {}", class.id);
            assert_eq!(class.parents, other.parents, "parents of {}", class.id);
        }
        let listed = |g: &G| -> Vec<(String, Vec<Id>)> {
            let mut v: Vec<_> = g
                .classes_by_op
                .iter()
                .filter(|(_, ids)| !ids.is_empty())
                .map(|(op, _)| (format!("{:?}", op), g.classes_for_op(op).unwrap().collect()))
                .collect();
            v.sort();
            v
        };
        assert_eq!(listed(a), listed(b), "classes_for_op");
        for g in [a, b] {
            assert!(g.clean);
            assert!(g.pending.is_empty() && g.analysis_pending.is_empty());
            assert!(g.dirty.is_empty() && g.unindexed.is_empty() && !g.reindex_all);
        }
    }

    /// A small deterministic generator (xorshift64*).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[derive(Clone, Debug)]
    enum Op {
        Add(Math),
        Union(Id, Id),
        Rebuild,
        /// `egraph[id].data = d`
        SetData(Id, Option<i32>),
        /// `set_analysis_data`
        SetAnalysis(Id, Option<i32>),
        /// Prune a node through `egraph[id]`, as an `Analysis::modify` may.
        Prune(Id),
        /// Borrow every class mutably.
        ClassesMut,
    }

    fn apply(g: &mut G, op: &Op) {
        match op {
            Op::Add(n) => {
                g.add(n.clone());
            }
            Op::Union(a, b) => {
                g.union(*a, *b);
            }
            Op::Rebuild => {
                g.rebuild();
            }
            Op::SetData(id, d) => g[*id].data = *d,
            Op::SetAnalysis(id, d) => g.set_analysis_data(*id, *d),
            Op::Prune(id) => {
                let class = &mut g[*id];
                if class.nodes.len() > 1 {
                    class.nodes.pop();
                }
            }
            Op::ClassesMut => g.classes_mut().for_each(|_| ()),
        }
    }

    fn random_op(rng: &mut Rng, g: &G) -> Op {
        let n = g.unionfind.size();
        let id = |rng: &mut Rng| Id::from(rng.below(n));
        match rng.below(14) {
            0..=1 => Op::Add(Math::Num(rng.below(8) as i32)),
            2 => Op::Add(Math::Symbol(format!("x{}", rng.below(12)).into())),
            3..=4 => Op::Add(Math::F([id(rng)])),
            5 => Op::Add(Math::Add([id(rng), id(rng)])),
            6..=7 => Op::Union(id(rng), id(rng)),
            8 => Op::Rebuild,
            9 => Op::SetData(id(rng), None),
            10 => Op::SetAnalysis(id(rng), Some(rng.below(8) as i32)),
            11 => Op::Prune(id(rng)),
            12 if rng.below(4) == 0 => Op::ClassesMut,
            _ => Op::Union(id(rng), id(rng)),
        }
    }

    /// Random adds, unions, rebuilds, data edits (through `egraph[id]` and
    /// `set_analysis_data`, both of which leave work pending without clearing
    /// `clean`), node pruning and `classes_mut` under random nested checkpoints,
    /// closed by rolling back a random open one (not only the innermost) or
    /// committing the innermost. Every rollback must restore exactly the e-graph as
    /// it was when its checkpoint was opened, iteration orders included, which must
    /// also be what replaying the operations before the checkpoint into a fresh
    /// e-graph produces. A clone taken under a checkpoint must have none and record
    /// nothing.
    #[test]
    fn rollback_restores_the_checkpointed_egraph() {
        let mut seen = [0usize; 11];
        for seed in 1..=400u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
            let mut g = G::default();
            let mut ops: Vec<Op> = Vec::new();
            for i in 0..4 {
                let op = Op::Add(Math::Num(i));
                apply(&mut g, &op);
                ops.push(op);
            }
            // (checkpoint, snapshot, number of operations before it)
            let mut open: Vec<(Checkpoint, G, usize)> = Vec::new();
            for _ in 0..200 {
                match rng.below(12) {
                    0 if open.len() < 4 => {
                        ops.push(Op::Rebuild);
                        let checkpoint = g.checkpoint();
                        open.push((checkpoint, g.clone(), ops.len()));
                    }
                    1 if !open.is_empty() => {
                        let k = rng.below(open.len());
                        let (checkpoint, snapshot, len) = open.drain(k..).next().unwrap();
                        for undo in &g.trail.as_ref().unwrap().log {
                            seen[match undo {
                                Undo::Memo { .. } => 0,
                                Undo::Listed { .. } => 1,
                                Undo::ParentPush { .. } => 2,
                                Undo::Data { .. } => 3,
                                Undo::Union { nodes, .. } => 4 + nodes.loser_first as usize,
                                Undo::Rebuilt { delta, .. } => {
                                    6 + !delta.dropped.is_empty() as usize
                                }
                                Undo::Class { .. } => 8,
                                Undo::Seen { .. } => unreachable!(),
                            }] += 1;
                        }
                        seen[9] += (checkpoint.depth + 1 < g.open_checkpoints()) as usize;
                        g.rollback(checkpoint);
                        ops.truncate(len);
                        assert_identical(&g, &snapshot);
                        let mut fresh = G::default();
                        ops.iter().for_each(|op| apply(&mut fresh, op));
                        fresh.rebuild();
                        assert_identical(&g, &fresh);
                    }
                    2 if !open.is_empty() => {
                        // keep the changes; the enclosing checkpoint still undoes them
                        let (checkpoint, _, _) = open.pop().unwrap();
                        seen[10] += 1;
                        g.commit(checkpoint);
                    }
                    3 if !open.is_empty() => {
                        let mut copy = g.clone();
                        assert_eq!(copy.open_checkpoints(), 0);
                        copy.add(Math::Symbol("new".into()));
                        let some = Id::from(rng.below(copy.unionfind.size()));
                        copy.union(some, Id::from(0));
                        copy.rebuild();
                        assert!(copy.trail.is_none() && copy.unionfind.log_len() == 0);
                    }
                    _ => {
                        let op = random_op(&mut rng, &g);
                        apply(&mut g, &op);
                        ops.push(op);
                    }
                }
                assert_eq!(g.open_checkpoints(), open.len());
            }
            while let Some((checkpoint, snapshot, _)) = open.pop() {
                g.rollback(checkpoint);
                assert_identical(&g, &snapshot);
            }
            assert!(g.trail.is_none());
            g.rebuild();
            assert!(g.check_classes() && g.check_memo());
        }
        // every kind of record, a dedup, an inner rollback and a commit happened
        assert!(seen.iter().all(|&n| n > 0), "{:?}", seen);
    }

    /// Work queued without clearing `clean` (by `set_analysis_data`, a mutable
    /// borrow of a class, `classes_mut`) is settled by `checkpoint`, not lost by a
    /// rollback: the result equals the same edits without a checkpoint.
    #[test]
    fn checkpoint_settles_pending_work() {
        let edits: [fn(&mut G, Id, Id); 3] = [
            |g, a, _| g.set_analysis_data(a, Some(2)),
            |g, _, fx| {
                let x = g.find(fx);
                g[x].nodes.retain(|n| matches!(n, Math::Symbol(_)));
            },
            |g, a, _| {
                for c in g.classes_mut() {
                    if c.id == a {
                        c.data = Some(5);
                    }
                }
            },
        ];
        for edit in edits {
            let mut g = G::default();
            let a = g.add(Math::Symbol("a".into()));
            let one = g.add(Math::Num(1));
            g.add(Math::Add([a, one]));
            let x = g.add(Math::Symbol("x".into()));
            let fx = g.add(Math::F([x]));
            g.union(x, fx);
            g.rebuild();
            edit(&mut g, a, fx);
            let mut reference = g.clone();
            reference.rebuild();
            let checkpoint = g.checkpoint();
            g.add(Math::Num(7));
            g.rebuild();
            g.rollback(checkpoint);
            assert_identical(&g, &reference);
        }
    }

    thread_local! {
        /// How often `checkpoint` had to rebuild.
        pub(super) static CHECKPOINT_REBUILDS: std::cell::Cell<usize> =
            const { std::cell::Cell::new(0) };
    }

    /// `checkpoint` rebuilds only when work is pending: on a rebuilt e-graph it
    /// costs nothing (a debug rebuild checks every class and memo entry).
    #[test]
    fn checkpoint_on_a_clean_egraph_does_not_rebuild() {
        let mut g = G::default();
        let x = g.add(Math::Symbol("x".into()));
        g.rebuild();
        let rebuilds = || CHECKPOINT_REBUILDS.with(|n| n.get());
        let before = rebuilds();
        let checkpoint = g.checkpoint();
        g.rollback(checkpoint);
        assert_eq!(rebuilds(), before);
        g.set_analysis_data(x, Some(1));
        let checkpoint = g.checkpoint();
        g.commit(checkpoint);
        assert_eq!(rebuilds(), before + 1);
    }

    /// Data a `merge` changes without reporting the change (e.g. a part of the
    /// data its ordering ignores) is restored all the same.
    #[test]
    fn silent_merge_is_restored() {
        #[derive(Default)]
        struct Silent;
        impl Analysis<Math> for Silent {
            /// (value, a tag `merge` updates silently)
            type Data = (Option<i32>, u32);
            fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
                to.1 = to.1.max(from.1);
                merge_max(&mut to.0, from.0)
            }
            fn make(egraph: &mut EGraph<Math, Self>, enode: &Math, _: Id) -> Self::Data {
                match enode {
                    Math::Num(n) => (Some(*n), 0),
                    Math::Symbol(_) => (Some(5), 7),
                    Math::F([a]) => (None, egraph[*a].data.1 + 1),
                    _ => (None, 0),
                }
            }
        }
        let mut g = EGraph::<Math, Silent>::default();
        let one = g.add(Math::Num(1));
        let x = g.add(Math::Symbol("x".into()));
        let f1 = g.add(Math::F([one]));
        g.rebuild();
        let before = (g[one].data, g[f1].data);
        let checkpoint = g.checkpoint();
        g.union(one, x);
        g.rebuild();
        // the parent's re-made data raised the tag without `merge` reporting it
        assert_eq!(g[f1].data, (None, 8));
        g.rollback(checkpoint);
        assert_eq!((g[one].data, g[f1].data), before);
    }

    /// A token closed by an enclosing rollback cannot close the checkpoint that
    /// later took its place.
    #[test]
    #[should_panic(expected = "not open on this e-graph")]
    fn stale_checkpoint_is_rejected() {
        let mut g = G::default();
        g.add(Math::Num(1));
        let outer = g.checkpoint();
        let inner = g.checkpoint();
        g.rollback(outer);
        let _again = g.checkpoint();
        let _again_inner = g.checkpoint();
        g.rollback(inner);
    }

    /// A token closes only the e-graph it was issued by.
    #[test]
    #[should_panic(expected = "not open on this e-graph")]
    fn foreign_checkpoint_is_rejected() {
        let mut g = G::default();
        let mut h = G::default();
        let checkpoint = g.checkpoint();
        let _other = h.checkpoint();
        h.commit(checkpoint);
    }

    /// Rolling back to an outer checkpoint closes the inner ones too.
    #[test]
    fn rollback_closes_inner_checkpoints() {
        let mut g = G::default();
        let x = g.add(Math::Symbol("x".into()));
        let y = g.add(Math::Symbol("y".into()));
        let snapshot = {
            g.rebuild();
            g.clone()
        };
        let outer = g.checkpoint();
        g.union(x, y);
        let _inner = g.checkpoint();
        g.add(Math::F([x]));
        g.rebuild();
        assert_eq!(g.open_checkpoints(), 2);
        g.rollback(outer);
        assert_eq!(g.open_checkpoints(), 0);
        assert_identical(&g, &snapshot);
    }

    /// Merging a small class into a large one records only the small class's
    /// nodes, never a copy of the large one: the trail grows with the change.
    #[test]
    fn trail_records_the_change_not_the_class() {
        let mut g = G::default();
        let hub = g.add(Math::Symbol("hub".into()));
        for i in 0..1000 {
            let s = g.add(Math::Symbol(format!("s{i}").into()));
            let fs = g.add(Math::F([s]));
            g.union(hub, fs);
        }
        g.rebuild();
        let hub = g.find(hub);
        assert!(g[hub].len() > 1000);
        let snapshot = g.clone();

        let checkpoint = g.checkpoint();
        let small = g.add(Math::Symbol("small".into()));
        g.union(hub, small);
        g.rebuild();
        let recorded: usize = g
            .trail
            .as_ref()
            .unwrap()
            .log
            .iter()
            .map(|undo| match undo {
                Undo::Union { .. } => 0,
                Undo::Rebuilt { delta, .. } => {
                    delta.canonicalized.len() + delta.moved.len() + delta.dropped.len()
                }
                Undo::Class { nodes, .. } => nodes.len(),
                Undo::Memo { .. } | Undo::Listed { .. } => 1,
                Undo::ParentPush { .. } | Undo::Data { .. } | Undo::Seen { .. } => 0,
            })
            .sum();
        assert!(recorded < 10, "recorded {} nodes", recorded);
        g.rollback(checkpoint);
        assert_identical(&g, &snapshot);
    }

    /// A large class losing a union to a small one (the small one has more
    /// parents) is not copied either: the record keeps list lengths.
    #[test]
    fn large_loser_is_not_copied() {
        let mut g = G::default();
        let hub = g.add(Math::Symbol("hub".into()));
        for i in 0..1000 {
            let s = g.add(Math::Symbol(format!("s{}", i).into()));
            let fs = g.add(Math::F([s]));
            g.union(hub, fs);
        }
        let small = g.add(Math::Symbol("small".into()));
        g.add(Math::F([small]));
        g.rebuild();
        let snapshot = g.clone();

        let checkpoint = g.checkpoint();
        g.union(small, hub);
        g.rebuild();
        assert_eq!(g.find(hub), g.find(small));
        let recorded: usize = g
            .trail
            .as_ref()
            .unwrap()
            .log
            .iter()
            .map(|undo| match undo {
                Undo::Rebuilt { delta, .. } => {
                    delta.canonicalized.len() + delta.moved.len() + delta.dropped.len()
                }
                Undo::Class { nodes, .. } => nodes.len(),
                _ => 0,
            })
            .sum();
        assert!(recorded < 10, "recorded {} nodes", recorded);
        g.rollback(checkpoint);
        assert_identical(&g, &snapshot);
    }

    #[test]
    fn no_trail_without_a_checkpoint() {
        let mut g = G::default();
        let x = g.add(Math::Num(1));
        let checkpoint = g.checkpoint();
        g.commit(checkpoint);
        let y = g.add(Math::Num(2));
        g.union(x, y);
        g.rebuild();
        assert!(g.trail.is_none());
        assert_eq!(g.unionfind.log_len(), 0);
    }

    #[test]
    #[should_panic(expected = "explanations")]
    fn checkpoints_reject_explanations() {
        let mut g = EGraph::<SymbolLang, ()>::default().with_explanations_enabled();
        let _ = g.checkpoint();
    }
}
