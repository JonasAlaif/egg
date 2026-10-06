use crate::Id;
use std::fmt::Debug;

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize, serde::Deserialize))]
pub struct UnionFind {
    parents: Vec<Id>,
    /// While a checkpoint is open: the previous parent of every overwritten entry
    /// below `logged_below` (entries at or above it are discarded by a rollback).
    #[cfg_attr(feature = "serde-1", serde(skip))]
    undo: Vec<(Id, Id)>,
    #[cfg_attr(feature = "serde-1", serde(skip))]
    logged_below: usize,
}

impl UnionFind {
    /// Records overwrites of the entries that exist now, until the next call;
    /// `0` stops recording.
    pub(crate) fn log_writes_below(&mut self, len: usize) {
        self.logged_below = len;
    }

    /// The length of the overwrite log, a position to [`rollback`](Self::rollback) to.
    pub(crate) fn log_len(&self) -> usize {
        self.undo.len()
    }

    /// Undoes the overwrites recorded since `log_len`, then forgets the sets made
    /// after the first `size` ones.
    pub(crate) fn rollback(&mut self, log_len: usize, size: usize) {
        for (id, parent) in self.undo.drain(log_len..).rev() {
            self.parents[usize::from(id)] = parent;
        }
        self.parents.truncate(size);
    }

    #[cfg(test)]
    pub(crate) fn parents(&self) -> &[Id] {
        &self.parents
    }

    /// Forgets the overwrite log (no checkpoint is open any more).
    pub(crate) fn clear_log(&mut self) {
        self.undo.clear();
        self.logged_below = 0;
    }

    fn set_parent(&mut self, query: Id, parent: Id) {
        let slot = &mut self.parents[usize::from(query)];
        if usize::from(query) < self.logged_below {
            self.undo.push((query, *slot));
        }
        *slot = parent;
    }

    pub fn make_set(&mut self) -> Id {
        let id = Id::from(self.parents.len());
        self.parents.push(id);
        id
    }

    pub fn size(&self) -> usize {
        self.parents.len()
    }

    fn parent(&self, query: Id) -> Id {
        self.parents[usize::from(query)]
    }

    pub fn find(&self, mut current: Id) -> Id {
        while current != self.parent(current) {
            current = self.parent(current)
        }
        current
    }

    pub fn find_mut(&mut self, mut current: Id) -> Id {
        while current != self.parent(current) {
            let grandparent = self.parent(self.parent(current));
            self.set_parent(current, grandparent);
            current = grandparent;
        }
        current
    }

    /// Given two leader ids, unions the two eclasses making root1 the leader.
    pub fn union(&mut self, root1: Id, root2: Id) -> Id {
        self.set_parent(root2, root1);
        root1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(us: impl IntoIterator<Item = usize>) -> Vec<Id> {
        us.into_iter().map(|u| u.into()).collect()
    }

    #[test]
    fn union_find() {
        let n = 10;
        let id = Id::from;

        let mut uf = UnionFind::default();
        for _ in 0..n {
            uf.make_set();
        }

        // test the initial condition of everyone in their own set
        assert_eq!(uf.parents, ids(0..n));

        // build up one set
        uf.union(id(0), id(1));
        uf.union(id(0), id(2));
        uf.union(id(0), id(3));

        // build up another set
        uf.union(id(6), id(7));
        uf.union(id(6), id(8));
        uf.union(id(6), id(9));

        // this should compress all paths
        for i in 0..n {
            uf.find_mut(id(i));
        }

        // indexes:         0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        let expected = vec![0, 0, 0, 0, 4, 5, 6, 6, 6, 6];
        assert_eq!(uf.parents, ids(expected));
    }
}
