use std::fmt::Debug;
use std::iter::ExactSizeIterator;

use crate::*;

/// An equivalence class of enodes.
#[non_exhaustive]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize, serde::Deserialize))]
pub struct EClass<L, D> {
    /// This eclass's id.
    pub id: Id,
    /// The equivalent enodes in this equivalence class.
    pub nodes: Vec<L>,
    /// The analysis data associated with this eclass.
    ///
    /// Modifying this field will _not_ cause changes to propagate through the e-graph.
    /// Prefer [`EGraph::set_analysis_data`] instead.
    pub data: D,
    /// The original Ids of parent enodes.
    pub(crate) parents: Vec<Id>,
}

/// The e-classes of an e-graph, indexed by their (canonical) id.
///
/// A dense vector rather than a hash map: lookups need no hashing, and iteration
/// visits classes in id order, so it depends only on which classes exist, never
/// on the order they were inserted or removed in.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct ClassMap<L, D> {
    slots: Vec<Option<EClass<L, D>>>,
    len: usize,
}

impl<L, D> Default for ClassMap<L, D> {
    fn default() -> Self {
        ClassMap {
            slots: Vec::new(),
            len: 0,
        }
    }
}

impl<L, D> ClassMap<L, D> {
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn get(&self, id: &Id) -> Option<&EClass<L, D>> {
        self.slots.get(usize::from(*id)).and_then(Option::as_ref)
    }

    pub(crate) fn get_mut(&mut self, id: &Id) -> Option<&mut EClass<L, D>> {
        self.slots
            .get_mut(usize::from(*id))
            .and_then(Option::as_mut)
    }

    pub(crate) fn insert(&mut self, id: Id, class: EClass<L, D>) -> Option<EClass<L, D>> {
        let i = usize::from(id);
        if self.slots.len() <= i {
            self.slots.resize_with(i + 1, || None);
        }
        let old = self.slots[i].replace(class);
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    pub(crate) fn remove(&mut self, id: &Id) -> Option<EClass<L, D>> {
        let old = self.slots.get_mut(usize::from(*id)).and_then(Option::take);
        if old.is_some() {
            self.len -= 1;
        }
        old
    }

    /// Removes every class with an id of at least `size` (all of them were created
    /// after the first `size` ids).
    pub(crate) fn truncate(&mut self, size: usize) {
        if size < self.slots.len() {
            self.len -= self.slots[size..].iter().filter(|s| s.is_some()).count();
            self.slots.truncate(size);
        }
    }

    pub(crate) fn values(&self) -> impl ExactSizeIterator<Item = &EClass<L, D>> {
        Counted {
            inner: self.slots.iter().flatten(),
            left: self.len,
        }
    }

    pub(crate) fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut EClass<L, D>> {
        Counted {
            inner: self.slots.iter_mut().flatten(),
            left: self.len,
        }
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &Id> {
        self.values().map(|class| &class.id)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&Id, &EClass<L, D>)> {
        self.values().map(|class| (&class.id, class))
    }

    /// Converts every class with `f`, keeping ids.
    pub(crate) fn map<L2, D2>(
        self,
        f: impl Fn(EClass<L, D>) -> EClass<L2, D2>,
    ) -> ClassMap<L2, D2> {
        ClassMap {
            slots: self.slots.into_iter().map(|s| s.map(&f)).collect(),
            len: self.len,
        }
    }
}

impl<L, D> std::ops::Index<&Id> for ClassMap<L, D> {
    type Output = EClass<L, D>;
    fn index(&self, id: &Id) -> &EClass<L, D> {
        self.get(id)
            .unwrap_or_else(|| panic!("Invalid class id {}", id))
    }
}

/// An iterator that knows how many items it has left.
struct Counted<I> {
    inner: I,
    left: usize,
}

impl<I: Iterator> Iterator for Counted<I> {
    type Item = I::Item;
    fn next(&mut self) -> Option<I::Item> {
        let next = self.inner.next();
        if next.is_some() {
            self.left -= 1;
        }
        next
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl<I: Iterator> ExactSizeIterator for Counted<I> {}

impl<L, D> EClass<L, D> {
    /// Returns `true` if the `eclass` is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the number of enodes in this eclass.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Iterates over the enodes in this eclass.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &L> {
        self.nodes.iter()
    }

    /// Iterates over the non-canonical ids of parent enodes of this eclass.
    pub fn parents(&self) -> impl ExactSizeIterator<Item = Id> + '_ {
        self.parents.iter().copied()
    }
}

impl<L: Language, D> EClass<L, D> {
    /// Iterates over the childless enodes in this eclass.
    pub fn leaves(&self) -> impl Iterator<Item = &L> {
        self.nodes.iter().filter(|&n| n.is_leaf())
    }

    /// Asserts that the childless enodes in this eclass are unique.
    pub fn assert_unique_leaves(&self)
    where
        L: Language,
    {
        let mut leaves = self.leaves();
        if let Some(first) = leaves.next() {
            assert!(
                leaves.all(|l| l == first),
                "Different leaves in eclass {}: {:?}",
                self.id,
                self.leaves().collect::<crate::util::HashSet<_>>()
            );
        }
    }

    /// Run some function on each matching e-node in this class.
    pub fn for_each_matching_node<Err>(
        &self,
        node: &L,
        mut f: impl FnMut(&L) -> Result<(), Err>,
    ) -> Result<(), Err>
    where
        L: Language,
    {
        if self.nodes.len() < 50 {
            self.nodes
                .iter()
                .filter(|n| node.matches(n))
                .try_for_each(f)
        } else {
            debug_assert!(node.all(|id| id == Id::from(0)));
            debug_assert!(self.nodes.windows(2).all(|w| w[0] < w[1]));
            let mut start = self.nodes.binary_search(node).unwrap_or_else(|i| i);
            let discrim = node.discriminant();
            while start > 0 {
                if self.nodes[start - 1].discriminant() == discrim {
                    start -= 1;
                } else {
                    break;
                }
            }
            let mut matching = self.nodes[start..]
                .iter()
                .take_while(|&n| n.discriminant() == discrim)
                .filter(|n| node.matches(n));
            debug_assert_eq!(
                matching.clone().count(),
                self.nodes.iter().filter(|n| node.matches(n)).count(),
                "matching node {:?}\nstart={}\n{:?} != {:?}\nnodes: {:?}",
                node,
                start,
                matching.clone().collect::<HashSet<_>>(),
                self.nodes
                    .iter()
                    .filter(|n| node.matches(n))
                    .collect::<HashSet<_>>(),
                self.nodes
            );
            matching.try_for_each(&mut f)
        }
    }
}
