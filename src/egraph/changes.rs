//! Change tracking: what changed in an [`EGraph`] since a point in time, for
//! incremental (semi-naive) e-matching.
//!
//! Once [`EGraph::track_changes`] is called the e-graph appends to a *change log*:
//! every e-node that is new to its e-class (added, re-canonicalized because a child
//! was merged away, or moved into the class by a union) and every e-class whose
//! analysis data changed. A match that exists now but did not exist at some earlier
//! point involves at least one of these, so a searcher that only looks at matches
//! involving a logged change finds every new match.
//!
//! A *subscriber* (for example a rewrite rule, named by a [`Symbol`]) records how far
//! it has consumed the log with [`EGraph::mark_seen`]. Subscriber positions live in
//! the e-graph so that a [`Checkpoint`](super::Checkpoint) rollback restores them
//! together with the log: work that was consumed after the checkpoint is pending
//! again. The log keeps what some subscriber has not consumed yet; with no
//! subscriber nothing is retained beyond the current position.

use super::*;

/// A position in the change log of an [`EGraph`]; see [`EGraph::track_changes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChangePos(usize);

/// One logged change.
#[derive(Clone, Debug)]
pub(crate) enum Change<L> {
    /// The e-node with this id was added, or its children were canonicalized again.
    Node(Id),
    /// A union moved this e-node into the class of the id.
    Moved(Id, L),
    /// The analysis data of this class changed.
    Data(Id),
    /// The class was borrowed mutably: any of its nodes and its data may have changed.
    Class(Id),
}

/// The change log of an [`EGraph`] and the position of each subscriber in it.
#[derive(Debug)]
pub(crate) struct ChangeLog<L> {
    /// The position of `entries[0]`.
    base: usize,
    entries: Vec<Change<L>>,
    seen: HashMap<Symbol, usize>,
    /// While a checkpoint is open: the lowest subscriber position when the
    /// outermost one was opened. A rollback may restore any position since.
    pub(crate) pinned: Option<usize>,
    /// Forgetting is tried again once the log is this long.
    trim_at: usize,
}

/// A clone has no open checkpoints, so nothing pinned.
impl<L: Clone> Clone for ChangeLog<L> {
    fn clone(&self) -> Self {
        ChangeLog {
            base: self.base,
            entries: self.entries.clone(),
            seen: self.seen.clone(),
            pinned: None,
            trim_at: self.trim_at,
        }
    }
}

impl<L> ChangeLog<L> {
    pub(crate) fn push(&mut self, change: Change<L>) {
        self.entries.push(change);
        // Forget the prefix no one needs, each time the log has doubled.
        if self.entries.len() >= self.trim_at {
            let drop = self.lowest_needed() - self.base;
            self.entries.drain(..drop);
            self.base += drop;
            self.trim_at = 2 * self.entries.len().max(64);
        }
    }

    pub(crate) fn end(&self) -> usize {
        self.base + self.entries.len()
    }

    /// The lowest position a subscriber can still ask for, now or after a rollback.
    pub(crate) fn lowest_needed(&self) -> usize {
        let lowest = self.seen.values().copied().min().unwrap_or(self.end());
        self.pinned.map_or(lowest, |pinned| pinned.min(lowest))
    }

    pub(crate) fn seen(&self, subscriber: Symbol) -> Option<usize> {
        self.seen.get(&subscriber).copied()
    }

    /// Sets the position of `subscriber` (`None`: forget it), returning the old one.
    pub(crate) fn set_seen(&mut self, subscriber: Symbol, pos: Option<usize>) -> Option<usize> {
        match pos {
            Some(pos) => self.seen.insert(subscriber, pos),
            None => self.seen.remove(&subscriber),
        }
    }

    pub(crate) fn subscribers(&self) -> Vec<Symbol> {
        let mut subscribers: Vec<Symbol> = self.seen.keys().copied().collect();
        subscribers.sort_unstable_by_key(|s| s.as_str());
        subscribers
    }

    /// Discards the entries from position `end` on.
    pub(crate) fn truncate(&mut self, end: usize) {
        self.entries.truncate(end - self.base);
    }
}

/// The changes made to an [`EGraph`] between a [`ChangePos`] and now, as canonical
/// `(e-class, e-node)` pairs grouped by operator and as the e-classes whose data
/// changed; returned by [`EGraph::changes_since`]. Every list is sorted and free of
/// duplicates, so iterating it does not depend on how the e-graph stores anything.
#[derive(Debug)]
pub struct Changes<L: Language> {
    /// Unique among all change sets (see [`Changes::serial`]).
    serial: u64,
    /// Sorted by class, then node.
    nodes: Vec<(Id, L)>,
    /// For each operator, the positions in `nodes` of its e-nodes.
    by_op: HashMap<L::Discriminant, Vec<usize>>,
    classes: Vec<Id>,
    data: Vec<Id>,
}

impl<L: Language> Changes<L> {
    /// The e-nodes that are new to their e-class: added, canonicalized again
    /// because a child was merged, or moved in by a union; as `(class, node)`.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &(Id, L)> {
        self.nodes.iter()
    }

    /// The changed e-nodes (see [`iter`](Changes::iter)) with operator `op`.
    pub fn nodes<'a>(&'a self, op: &L::Discriminant) -> impl Iterator<Item = &'a (Id, L)> {
        let positions = self.by_op.get(op).map_or(&[][..], |p| p);
        positions.iter().map(move |&i| &self.nodes[i])
    }

    /// The e-classes holding a changed e-node.
    pub fn classes(&self) -> &[Id] {
        &self.classes
    }

    /// The e-classes whose analysis data changed.
    pub fn data(&self) -> &[Id] {
        &self.data
    }

    /// A number no other change set has, so searchers can share what they derive
    /// from this one (a rule set with many similar searchers walks the changes once).
    pub fn serial(&self) -> u64 {
        self.serial
    }

    /// Whether nothing changed.
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty() && self.data.is_empty()
    }
}

impl<L: Language, N: Analysis<L>> EGraph<L, N> {
    /// Starts logging changes, for incremental e-matching (see [`Changes`]); a
    /// no-op if they are logged already. Without this the e-graph keeps no log
    /// and pays nothing.
    pub fn track_changes(&mut self) {
        self.changes.get_or_insert_with(|| {
            Box::new(ChangeLog {
                base: 0,
                entries: vec![],
                seen: HashMap::default(),
                pinned: None,
                trim_at: 0,
            })
        });
    }

    /// The current end of the change log, or `None` if changes are not tracked.
    pub fn change_pos(&self) -> Option<ChangePos> {
        self.changes.as_ref().map(|log| ChangePos(log.end()))
    }

    /// The position up to which `subscriber` has consumed the changes, or `None`
    /// if it has not consumed any (it must look at the whole e-graph).
    pub fn seen(&self, subscriber: impl Into<Symbol>) -> Option<ChangePos> {
        let log = self.changes.as_ref()?;
        log.seen(subscriber.into()).map(ChangePos)
    }

    /// Records that `subscriber` has consumed the changes up to `pos`. Changes
    /// before the position of every subscriber are forgotten. A
    /// [rollback](EGraph::rollback) restores the position the subscriber had when
    /// the checkpoint was opened.
    pub fn mark_seen(&mut self, subscriber: impl Into<Symbol>, pos: ChangePos) {
        let log = self.changes.as_mut().expect("changes are not tracked");
        debug_assert!(pos.0 <= log.end());
        let subscriber = subscriber.into();
        let old = log.set_seen(subscriber, Some(pos.0));
        if let Some(trail) = &mut *self.trail {
            trail.seen(subscriber, old);
        }
    }

    /// The changes made since `pos`, or `None` if the log no longer holds them
    /// (some subscriber's position must be at or before `pos` to keep them). The
    /// e-graph must be clean.
    pub fn changes_since(&self, pos: ChangePos) -> Option<Changes<L>> {
        assert!(self.clean, "changes of a dirty e-graph");
        let log = self.changes.as_ref()?;
        let start = pos.0.checked_sub(log.base)?;
        let canonical = |node: &L| node.clone().map_children(|c| self.find(c));
        let mut nodes: Vec<(Id, L)> = vec![];
        let mut data = vec![];
        for change in &log.entries[start..] {
            match change {
                Change::Node(id) => {
                    nodes.push((self.find(*id), canonical(&self.nodes[usize::from(*id)])))
                }
                Change::Moved(id, node) => nodes.push((self.find(*id), canonical(node))),
                Change::Data(id) => data.push(self.find(*id)),
                Change::Class(id) => {
                    let id = self.find(*id);
                    nodes.extend(self[id].nodes.iter().map(|n| (id, n.clone())));
                    data.push(id);
                }
            }
        }
        nodes.sort_unstable();
        nodes.dedup();
        data.sort_unstable();
        data.dedup();
        let mut classes: Vec<Id> = nodes.iter().map(|(c, _)| *c).collect();
        classes.sort_unstable();
        classes.dedup();
        let mut by_op: HashMap<L::Discriminant, Vec<usize>> = HashMap::default();
        for (i, (_, node)) in nodes.iter().enumerate() {
            by_op.entry(node.discriminant()).or_default().push(i);
        }
        static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Some(Changes {
            serial: SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            nodes,
            by_op,
            classes,
            data,
        })
    }

    /// The classes holding an e-node that matches `path` (from the top) down to one
    /// of `classes`: each step is an operator (an e-node whose children are
    /// ignored) and the child index the path descends through.
    pub(crate) fn ancestors(&self, mut classes: Vec<Id>, path: &[(L, usize)]) -> Vec<Id> {
        classes.iter_mut().for_each(|c| *c = self.find(*c));
        classes.sort_unstable();
        classes.dedup();
        let mut level = classes;
        for (op, i) in path.iter().rev() {
            let mut up = vec![];
            for &child in &level {
                for parent in self[child].parents() {
                    let node = &self.nodes[usize::from(parent)];
                    if op.matches(node) && self.find(node.children()[*i]) == child {
                        up.push(self.find(parent));
                    }
                }
            }
            up.sort_unstable();
            up.dedup();
            level = up;
        }
        level
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

    /// Constant folding whose `modify` adds and unions, and whose data can change
    /// more than once (`merge_max`), so data changes propagate through rebuilds.
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

    /// A match: its root, its substitution and the data of both. When it was
    /// reported its ids are canonicalized again, its data kept as it was.
    type Match = (Id, Vec<Id>, Vec<Option<i32>>);

    fn read(g: &G, root: Id, subst: &[Id]) -> Match {
        let root = g.find(root);
        let subst: Vec<Id> = subst.iter().map(|&id| g.find(id)).collect();
        let mut data = vec![g[root].data];
        data.extend(subst.iter().map(|&id| g[id].data));
        (root, subst, data)
    }

    /// The matches of `pattern` in `found`, as they read now.
    fn flatten(g: &G, pattern: &Pattern<Math>, found: &[SearchMatches<Math>]) -> Vec<Match> {
        let vars = pattern.vars();
        let mut out = vec![];
        for m in found {
            for subst in &m.substs {
                let ids: Vec<Id> = vars.iter().map(|v| subst[*v]).collect();
                out.push(read(g, m.eclass, &ids));
            }
        }
        out
    }

    /// Random adds, unions, data edits and rebuilds under random nested checkpoints,
    /// commits and rollbacks. Each pattern is a subscriber that searches only the
    /// changes since it last searched. Every match a full search finds must be one
    /// it reported before (as that match reads now: root, substitution and their
    /// data) or one it reports now. A rollback must restore what each pattern had
    /// seen when the checkpoint was opened, so the client forgets what it was told
    /// since.
    #[test]
    fn incremental_search_finds_every_new_match() {
        let patterns: Vec<Pattern<Math>> = [
            "(f ?x)",
            "(+ ?x ?y)",
            "(+ ?x ?x)",
            "(+ ?x 0)",
            "(f (f ?x))",
            "(+ (f ?x) ?y)",
            "(+ (f ?x) (f ?x))",
            "(+ ?x (+ ?y ?x))",
            "(+ (f ?x) ?x)",
            "(f (+ (f ?x) ?x))",
            "?x",
        ]
        .iter()
        .map(|p| p.parse().unwrap())
        .collect();
        let name = |i: usize| Symbol::from(format!("p{}", i));
        let (mut incremental, mut full) = (0, 0);
        let (mut rollbacks, mut commits) = (0, 0);
        for seed in 1..=200u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
            let mut g = G::default();
            g.track_changes();
            g.add(Math::Num(0));
            // what each pattern has reported
            let mut reported: Vec<Vec<Match>> = vec![vec![]; patterns.len()];
            let mut open: Vec<(Checkpoint, Vec<Vec<Match>>)> = vec![];
            for _ in 0..150 {
                let id = |rng: &mut Rng, g: &G| Id::from(rng.below(g.nodes.len()));
                match rng.below(14) {
                    0..=1 => {
                        g.add(Math::Num(rng.below(4) as i32));
                    }
                    2 => {
                        g.add(Math::Symbol(format!("x{}", rng.below(10)).into()));
                    }
                    3..=4 => {
                        let x = id(&mut rng, &g);
                        g.add(Math::F([x]));
                    }
                    5..=6 => {
                        let (x, y) = (id(&mut rng, &g), id(&mut rng, &g));
                        g.add(Math::Add([x, y]));
                    }
                    7 => {
                        let (x, y) = (id(&mut rng, &g), id(&mut rng, &g));
                        g.union(x, y);
                    }
                    8 => {
                        let x = id(&mut rng, &g);
                        g[x].data = None;
                    }
                    9 if open.len() < 3 => {
                        let checkpoint = g.checkpoint();
                        open.push((checkpoint, reported.clone()));
                    }
                    10 if !open.is_empty() => {
                        let (checkpoint, before) = open.pop().unwrap();
                        g.rollback(checkpoint);
                        reported = before;
                        rollbacks += 1;
                    }
                    11 if !open.is_empty() => {
                        let (checkpoint, _) = open.pop().unwrap();
                        g.commit(checkpoint);
                        commits += 1;
                    }
                    _ => {
                        g.rebuild();
                        let pos = g.change_pos().unwrap();
                        for (i, pattern) in patterns.iter().enumerate() {
                            let found = match g.seen(name(i)) {
                                Some(seen) => {
                                    let changes = g.changes_since(seen).unwrap();
                                    pattern.search_changes(&g, &changes, usize::MAX)
                                }
                                None => pattern.search(&g),
                            };
                            let found = flatten(&g, pattern, &found);
                            let all = flatten(&g, pattern, &pattern.search(&g));
                            incremental += found.len();
                            full += all.len();
                            reported[i].extend(found);
                            let known: HashSet<Match> = reported[i]
                                .iter()
                                .map(|(root, subst, data)| {
                                    let (root, subst, _) = read(&g, *root, subst);
                                    (root, subst, data.clone())
                                })
                                .collect();
                            for m in all {
                                assert!(
                                    known.contains(&m),
                                    "seed {}: {} missed {:?}",
                                    seed,
                                    pattern,
                                    m
                                );
                            }
                            g.mark_seen(name(i), pos);
                        }
                    }
                }
            }
        }
        assert!(rollbacks > 100 && commits > 100, "{} {}", rollbacks, commits);
        // and the search must actually be incremental
        assert!(incremental * 3 < full, "{} of {}", incremental, full);
    }

    /// Changes every subscriber has consumed are forgotten, and changes a rollback
    /// could ask for again are kept.
    #[test]
    fn consumed_changes_are_forgotten() {
        let mut g = G::default();
        g.track_changes();
        let log_len = |g: &G| g.changes.as_ref().unwrap().entries.len();
        for i in 0..1000 {
            g.add(Math::Num(i));
            g.rebuild();
            let pos = g.change_pos().unwrap();
            g.mark_seen("a", pos);
        }
        assert!(log_len(&g) <= 128, "{}", log_len(&g));

        let seen = g.change_pos().unwrap();
        let checkpoint = g.checkpoint();
        for i in 1000..2000 {
            g.add(Math::Num(i));
            g.rebuild();
            let pos = g.change_pos().unwrap();
            g.mark_seen("a", pos);
        }
        g.rollback(checkpoint);
        assert_eq!(g.seen("a"), Some(seen));
        assert!(g.changes_since(seen).unwrap().is_empty());
    }
}
