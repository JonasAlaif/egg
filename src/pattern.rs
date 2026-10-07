use fmt::Formatter;
use log::*;
use std::borrow::Cow;
use std::convert::TryInto;
use std::fmt::{self, Display};
use std::{convert::TryFrom, str::FromStr};

use thiserror::Error;

use crate::*;

/// A pattern that can function as either a [`Searcher`] or [`Applier`].
///
/// A [`Pattern`] is essentially a for-all quantified expression with
/// [`Var`]s as the variables (in the logical sense).
///
/// When creating a [`Rewrite`], the most common thing to use as either
/// the left hand side (the [`Searcher`]) or the right hand side
/// (the [`Applier`]) is a [`Pattern`].
///
/// As a [`Searcher`], a [`Pattern`] does the intuitive
/// thing.
/// Here is a somewhat verbose formal-ish statement:
/// Searching for a pattern in an egraph yields substitutions
/// ([`Subst`]s) _s_ such that, for any _s'_—where instead of
/// mapping a variables to an eclass as _s_ does, _s'_ maps
/// a variable to an arbitrary expression represented by that
/// eclass—_p[s']_ (the pattern under substitution _s'_) is also
/// represented by the egraph.
///
/// As an [`Applier`], a [`Pattern`] performs the given substitution
/// and adds the result to the [`EGraph`].
///
/// Importantly, [`Pattern`] implements [`FromStr`] if the
/// [`Language`] does.
/// This is probably how you'll create most [`Pattern`]s.
///
/// ```
/// use egg::*;
/// define_language! {
///     enum Math {
///         Num(i32),
///         "+" = Add([Id; 2]),
///     }
/// }
///
/// let mut egraph = EGraph::<Math, ()>::default();
/// let a11 = egraph.add_expr(&"(+ 1 1)".parse().unwrap());
/// let a22 = egraph.add_expr(&"(+ 2 2)".parse().unwrap());
///
/// // use Var syntax (leading question mark) to get a
/// // variable in the Pattern
/// let same_add: Pattern<Math> = "(+ ?a ?a)".parse().unwrap();
///
/// // Rebuild before searching
/// egraph.rebuild();
///
/// // This is the search method from the Searcher trait
/// let matches = same_add.search(&egraph);
/// let matched_eclasses: Vec<Id> = matches.iter().map(|m| m.eclass).collect();
/// // matches come in e-class id order
/// assert_eq!(matched_eclasses, vec![a11, a22]);
/// ```
///
/// [`FromStr`]: std::str::FromStr
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern<L> {
    /// The actual pattern as a [`RecExpr`]
    pub ast: PatternAst<L>,
    program: machine::Program<L>,
    places: Vec<Place<L>>,
    /// Whose analysis data a match's consumer reads: the root's, and the classes
    /// of these variables (`None`: all of them). See [`Pattern::with_data_reads`].
    data_reads: (bool, Option<Vec<Var>>),
}

/// An e-node or variable occurrence of a pattern, with the path to it from the
/// root: for each e-node above it, its operator (children zeroed) and the child
/// taken.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Place<L> {
    /// The operator of an e-node, `None` for a variable.
    op: Option<L>,
    var: Option<Var>,
    path: Vec<(L, usize)>,
    /// For an e-node below the root whose ancestors' other children are all
    /// variables of its own subpattern: the subpattern's matcher, and the
    /// ancestors bottom-up (each its node and the child taken). From a match of
    /// the subpattern at a changed e-node, each ancestor is then determined and
    /// found by a lookup instead of a scan of parents.
    climb: Option<(machine::Program<L>, Vec<Step<L>>)>,
}

/// An ancestor of a place: its operator (children zeroed), the child the path
/// takes, and the variable at each other child.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Step<L> {
    op: L,
    child: usize,
    vars: Vec<Option<Var>>,
}

impl<L: Language> Place<L> {
    fn all(ast: &PatternAst<L>) -> Vec<Self> {
        fn vars<L: Language>(ast: &PatternAst<L>, id: Id, out: &mut Vec<Var>) {
            match &ast[id] {
                ENodeOrVar::Var(v) => out.push(*v),
                ENodeOrVar::ENode(n) => n.for_each(|c| vars(ast, c, out)),
            }
        }
        fn walk<L: Language>(
            ast: &PatternAst<L>,
            id: Id,
            above: &mut Vec<(Id, usize)>,
            out: &mut Vec<Place<L>>,
        ) {
            let op_of = |id: Id| match &ast[id] {
                ENodeOrVar::ENode(n) => n.clone().map_children(|_| Id::from(0)),
                ENodeOrVar::Var(_) => unreachable!("a variable has no children"),
            };
            let path = above.iter().map(|&(a, i)| (op_of(a), i)).collect();
            match &ast[id] {
                ENodeOrVar::Var(v) => out.push(Place {
                    op: None,
                    var: Some(*v),
                    path,
                    climb: None,
                }),
                ENodeOrVar::ENode(node) => {
                    let mut bound = vec![];
                    vars(ast, id, &mut bound);
                    let groundable = above.iter().all(|&(a, i)| match &ast[a] {
                        ENodeOrVar::ENode(n) => n.children().iter().enumerate().all(|(j, &c)| {
                            j == i || matches!(&ast[c], ENodeOrVar::Var(v) if bound.contains(v))
                        }),
                        ENodeOrVar::Var(_) => false,
                    });
                    let climb = (!above.is_empty() && groundable).then(|| {
                        let sub = machine::Program::compile_from_pat(&ast.extract(id));
                        let steps = above.iter().rev().map(|&(a, i)| Step {
                            op: op_of(a),
                            child: i,
                            vars: match &ast[a] {
                                ENodeOrVar::ENode(n) => n
                                    .children()
                                    .iter()
                                    .map(|&c| match &ast[c] {
                                        ENodeOrVar::Var(v) => Some(*v),
                                        ENodeOrVar::ENode(_) => None,
                                    })
                                    .collect(),
                                ENodeOrVar::Var(_) => unreachable!("an ancestor is an e-node"),
                            },
                        });
                        (sub, steps.collect())
                    });
                    out.push(Place {
                        op: Some(op_of(id)),
                        var: None,
                        path,
                        climb,
                    });
                    for (i, &child) in node.children().iter().enumerate() {
                        above.push((id, i));
                        walk(ast, child, above, out);
                        above.pop();
                    }
                }
            }
        }
        let mut out = vec![];
        walk(ast, ast.root(), &mut vec![], &mut out);
        out
    }
}

/// A [`RecExpr`] that represents a
/// [`Pattern`].
pub type PatternAst<L> = RecExpr<ENodeOrVar<L>>;

impl<L: Language> PatternAst<L> {
    /// Returns a new `PatternAst` with the variables renames canonically
    pub fn alpha_rename(&self) -> Self {
        let mut vars = HashMap::<Var, Var>::default();
        let mut new = PatternAst::default();

        fn mkvar(i: usize) -> Var {
            let vs = &["?x", "?y", "?z", "?w"];
            match vs.get(i) {
                Some(v) => v.parse().unwrap(),
                None => format!("?v{}", i - vs.len()).parse().unwrap(),
            }
        }

        for n in self {
            new.add(match n {
                ENodeOrVar::ENode(_) => n.clone(),
                ENodeOrVar::Var(v) => {
                    let i = vars.len();
                    ENodeOrVar::Var(*vars.entry(*v).or_insert_with(|| mkvar(i)))
                }
            });
        }

        new
    }
}

impl<L: Language> Pattern<L> {
    /// Creates a new pattern from the given pattern ast.
    pub fn new(ast: PatternAst<L>) -> Self {
        let ast = ast.compact();
        let program = machine::Program::compile_from_pat(&ast);
        let places = Place::all(&ast);
        Pattern {
            ast,
            program,
            places,
            data_reads: (true, None),
        }
    }

    /// Declares whose analysis data the consumer of this pattern's matches reads:
    /// the root class's if `root`, and the classes bound to `vars`. By default it
    /// is all of them. [`Searcher::search_changes`] then returns a match whose
    /// data changed only if that data is read: a rewrite whose applier reads no
    /// data (a [`Pattern`] right-hand side) passes `false` and no variables.
    pub fn with_data_reads(mut self, root: bool, vars: &[Var]) -> Self {
        self.data_reads = (root, Some(vars.to_vec()));
        self
    }

    /// Returns a list of the [`Var`]s in this pattern.
    pub fn vars(&self) -> Vec<Var> {
        let mut vars = vec![];
        for n in &self.ast {
            if let ENodeOrVar::Var(v) = n {
                if !vars.contains(v) {
                    vars.push(*v)
                }
            }
        }
        vars
    }
}

impl<L: Language + Display> Pattern<L> {
    /// Pretty print this pattern as a sexp with the given width
    pub fn pretty(&self, width: usize) -> String {
        self.ast.pretty(width)
    }
}

/// The language of [`Pattern`]s.
///
#[derive(Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord)]
pub enum ENodeOrVar<L> {
    /// An enode from the underlying [`Language`]
    ENode(L),
    /// A pattern variable
    Var(Var),
}

/// The discriminant for the language of [`Pattern`]s.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum ENodeOrVarDiscriminant<L: Language> {
    ENode(L::Discriminant),
    Var(Var),
}

impl<L: Language> Language for ENodeOrVar<L> {
    type Discriminant = ENodeOrVarDiscriminant<L>;

    #[inline(always)]
    fn discriminant(&self) -> Self::Discriminant {
        match self {
            ENodeOrVar::ENode(n) => ENodeOrVarDiscriminant::ENode(n.discriminant()),
            ENodeOrVar::Var(v) => ENodeOrVarDiscriminant::Var(*v),
        }
    }

    fn matches(&self, _other: &Self) -> bool {
        panic!("Should never call this")
    }

    fn children(&self) -> &[Id] {
        match self {
            ENodeOrVar::ENode(n) => n.children(),
            ENodeOrVar::Var(_) => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            ENodeOrVar::ENode(n) => n.children_mut(),
            ENodeOrVar::Var(_) => &mut [],
        }
    }
}

impl<L: Language + Display> Display for ENodeOrVar<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ENode(node) => Display::fmt(node, f),
            Self::Var(var) => Display::fmt(var, f),
        }
    }
}

#[derive(Debug, Error)]
pub enum ENodeOrVarParseError<E> {
    #[error(transparent)]
    BadVar(<Var as FromStr>::Err),

    #[error("tried to parse pattern variable {0:?} as an operator")]
    UnexpectedVar(String),

    #[error(transparent)]
    BadOp(E),
}

impl<L: FromOp> FromOp for ENodeOrVar<L> {
    type Error = ENodeOrVarParseError<L::Error>;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        use ENodeOrVarParseError::*;

        if op.starts_with('?') && op.len() > 1 {
            if children.is_empty() {
                op.parse().map(Self::Var).map_err(BadVar)
            } else {
                Err(UnexpectedVar(op.to_owned()))
            }
        } else {
            L::from_op(op, children).map(Self::ENode).map_err(BadOp)
        }
    }
}

impl<L: FromOp> std::str::FromStr for Pattern<L> {
    type Err = RecExprParseError<ENodeOrVarParseError<L::Error>>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        PatternAst::from_str(s).map(Self::from)
    }
}

impl<'a, L: Language> From<&'a [L]> for Pattern<L> {
    fn from(expr: &'a [L]) -> Self {
        let ast = expr.iter().cloned().map(ENodeOrVar::ENode).collect();
        Self::new(ast)
    }
}

impl<L: Language> From<RecExpr<L>> for Pattern<L> {
    fn from(expr: RecExpr<L>) -> Self {
        let ast = expr.into_iter().map(ENodeOrVar::ENode).collect();
        Self::new(ast)
    }
}

impl<L: Language> From<&RecExpr<L>> for Pattern<L> {
    fn from(expr: &RecExpr<L>) -> Self {
        Self::from(expr.as_ref())
    }
}

impl<L: Language> From<PatternAst<L>> for Pattern<L> {
    fn from(ast: PatternAst<L>) -> Self {
        Self::new(ast)
    }
}

impl<L: Language> TryFrom<PatternAst<L>> for RecExpr<L> {
    type Error = Var;
    fn try_from(ast: PatternAst<L>) -> Result<Self, Self::Error> {
        ast.into_iter()
            .map(|n| match n {
                ENodeOrVar::ENode(n) => Ok(n),
                ENodeOrVar::Var(v) => Err(v),
            })
            .collect()
    }
}

impl<L: Language> TryFrom<Pattern<L>> for RecExpr<L> {
    type Error = Var;
    fn try_from(pat: Pattern<L>) -> Result<Self, Self::Error> {
        pat.ast.try_into()
    }
}

impl<L: Language + Display> Display for Pattern<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.ast, f)
    }
}

/// The result of searching a [`Searcher`] over one eclass.
///
/// Note that one [`SearchMatches`] can contain many found
/// substitutions. So taking the length of a list of [`SearchMatches`]
/// tells you how many eclasses something was matched in, _not_ how
/// many matches were found total.
///
#[derive(Debug)]
pub struct SearchMatches<'a, L: Language> {
    /// The eclass id that these matches were found in.
    pub eclass: Id,
    /// The substitutions for each match.
    pub substs: Vec<Subst>,
    /// Optionally, an ast for the matches used in proof production.
    pub ast: Option<Cow<'a, PatternAst<L>>>,
}

/// The root e-node (with its class) above `class` along `steps` (bottom-up: each
/// ancestor pattern node and the child taken), its other children read from
/// `subst`; `None` if some ancestor is not in the e-graph.
fn climb<L: Language, A: Analysis<L>>(
    egraph: &EGraph<L, A>,
    class: Id,
    subst: &Subst,
    steps: &[Step<L>],
) -> Option<(Id, L)> {
    let mut class = class;
    let mut node = None;
    for step in steps {
        let mut j = 0;
        let up = step.op.clone().map_children(|_| {
            let child = match step.vars[j] {
                _ if j == step.child => class,
                Some(v) => subst[v],
                None => unreachable!("a climbable ancestor's other children are variables"),
            };
            j += 1;
            egraph.find(child)
        });
        class = egraph.find(egraph.lookup(up.clone())?);
        node = Some(up);
    }
    node.map(|node| (class, node))
}

impl<L: Language, A: Analysis<L>> Searcher<L, A> for Pattern<L> {
    fn get_pattern_ast(&self) -> Option<&PatternAst<L>> {
        Some(&self.ast)
    }

    fn search_with_limit(&self, egraph: &EGraph<L, A>, limit: usize) -> Vec<SearchMatches<L>> {
        match self.ast.last().unwrap() {
            ENodeOrVar::ENode(e) => {
                let key = e.discriminant();
                match egraph.classes_for_op(&key) {
                    None => vec![],
                    Some(ids) => rewrite::search_eclasses_with_limit(self, egraph, ids, limit),
                }
            }
            ENodeOrVar::Var(_) => rewrite::search_eclasses_with_limit(
                self,
                egraph,
                egraph.classes().map(|e| e.id),
                limit,
            ),
        }
    }

    /// Searches only where a change can have made a new match: at each changed
    /// e-node in a position of the pattern (the root e-node itself; below the root,
    /// the root e-nodes found from it by lookups, or else the root classes above it
    /// along its path), and at every root whose match can bind a class whose data
    /// changed and is read (see [`Pattern::with_data_reads`]).
    fn search_changes(
        &self,
        egraph: &EGraph<L, A>,
        changes: &Changes<L>,
        mut limit: usize,
    ) -> Vec<SearchMatches<'_, L>> {
        let mut roots: Vec<Id> = vec![];
        let mut pinned: Vec<(Id, L)> = vec![];
        let (read_root, read_vars) = &self.data_reads;
        for place in &self.places {
            // the classes at this place of a match that may be new
            let mut bottoms: Vec<Id> = vec![];
            match &place.op {
                Some(op) => {
                    for (class, node) in changes.nodes(&op.discriminant()) {
                        if !op.matches(node) {
                            continue;
                        }
                        match &place.climb {
                            _ if place.path.is_empty() => pinned.push((*class, node.clone())),
                            Some((sub, steps)) => {
                                for subst in sub.run_at(egraph, *class, Some(node), usize::MAX) {
                                    pinned.extend(climb(egraph, *class, &subst, steps));
                                }
                            }
                            None => bottoms.push(*class),
                        }
                    }
                }
                // a variable root matches every class, so a new class is a new match
                None if place.path.is_empty() => roots.extend(changes.classes()),
                None => {}
            }
            let reads = match place.var {
                _ if place.path.is_empty() => *read_root,
                Some(v) => read_vars.as_ref().map_or(true, |vars| vars.contains(&v)),
                None => false,
            };
            if reads {
                bottoms.extend(changes.data());
            }
            if !bottoms.is_empty() {
                roots.extend(egraph.ancestors(bottoms, &place.path));
            }
        }
        roots.sort_unstable();
        roots.dedup();
        pinned.sort_unstable();
        pinned.dedup();
        pinned.retain(|(class, _)| roots.binary_search(class).is_err());

        let mut matches = vec![];
        for eclass in roots {
            if limit == 0 {
                break;
            }
            if let Some(m) = self.search_eclass_with_limit(egraph, eclass, limit) {
                limit -= m.substs.len();
                matches.push(m);
            }
        }
        for group in pinned.chunk_by(|a, b| a.0 == b.0) {
            let eclass = group[0].0;
            let mut substs = vec![];
            for (_, node) in group {
                substs.extend(self.program.run_at(egraph, eclass, Some(node), limit));
                limit -= substs.len().min(limit);
            }
            if !substs.is_empty() {
                let ast = Some(Cow::Borrowed(&self.ast));
                matches.push(SearchMatches {
                    eclass,
                    substs,
                    ast,
                });
            }
        }
        matches.sort_by_key(|m| m.eclass);
        matches
    }

    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<L, A>,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<L>> {
        let substs = self.program.run_with_limit(egraph, eclass, limit);
        if substs.is_empty() {
            None
        } else {
            let ast = Some(Cow::Borrowed(&self.ast));
            Some(SearchMatches {
                eclass,
                substs,
                ast,
            })
        }
    }

    fn vars(&self) -> Vec<Var> {
        Pattern::vars(self)
    }
}

impl<L, A> Applier<L, A> for Pattern<L>
where
    L: Language,
    A: Analysis<L>,
{
    fn get_pattern_ast(&self) -> Option<&PatternAst<L>> {
        Some(&self.ast)
    }

    fn apply_matches(
        &self,
        egraph: &mut EGraph<L, A>,
        matches: &[SearchMatches<L>],
        rule_name: Symbol,
    ) -> Vec<Id> {
        let mut added = vec![];
        let mut id_buf = vec![0.into(); self.ast.len()];
        for mat in matches {
            let sast = mat.ast.as_ref().map(|cow| cow.as_ref());
            for subst in &mat.substs {
                let did_something;
                let id;
                if egraph.are_explanations_enabled() {
                    let (id_temp, did_something_temp) =
                        egraph.union_instantiations(sast.unwrap(), &self.ast, subst, rule_name);
                    did_something = did_something_temp;
                    id = id_temp;
                } else {
                    id = apply_pat(&mut id_buf, &self.ast, egraph, subst);
                    did_something = egraph.union(id, mat.eclass);
                }

                if did_something {
                    added.push(id)
                }
            }
        }
        added
    }

    fn apply_one(
        &self,
        egraph: &mut EGraph<L, A>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<L>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let mut id_buf = vec![0.into(); self.ast.len()];
        let id = apply_pat(&mut id_buf, &self.ast, egraph, subst);

        if let Some(ast) = searcher_ast {
            let (from, did_something) =
                egraph.union_instantiations(ast, &self.ast, subst, rule_name);
            if did_something {
                vec![from]
            } else {
                vec![]
            }
        } else if egraph.union(eclass, id) {
            vec![eclass]
        } else {
            vec![]
        }
    }

    fn vars(&self) -> Vec<Var> {
        Pattern::vars(self)
    }
}

pub(crate) fn apply_pat<L: Language, A: Analysis<L>>(
    ids: &mut [Id],
    pat: &[ENodeOrVar<L>],
    egraph: &mut EGraph<L, A>,
    subst: &Subst,
) -> Id {
    debug_assert_eq!(pat.len(), ids.len());
    trace!("apply_rec {:2?} {:?}", pat, subst);

    for (i, pat_node) in pat.iter().enumerate() {
        let id = match pat_node {
            ENodeOrVar::Var(w) => subst[*w],
            ENodeOrVar::ENode(e) => {
                let n = e.clone().map_children(|child| ids[usize::from(child)]);
                trace!("adding: {:?}", n);
                egraph.add(n)
            }
        };
        ids[i] = id;
    }

    *ids.last().unwrap()
}

#[cfg(test)]
mod tests {

    use crate::{SymbolLang as S, *};

    type EGraph = crate::EGraph<S, ()>;

    #[test]
    fn simple_match() {
        crate::init_logger();
        let mut egraph = EGraph::default();

        let (plus_id, _) = egraph.union_instantiations(
            &"(+ x y)".parse().unwrap(),
            &"(+ z w)".parse().unwrap(),
            &Default::default(),
            "union_plus".to_string(),
        );
        egraph.rebuild();

        let commute_plus = rewrite!(
            "commute_plus";
            "(+ ?a ?b)" => "(+ ?b ?a)"
        );

        let matches = commute_plus.search(&egraph);
        let n_matches: usize = matches.iter().map(|m| m.substs.len()).sum();
        assert_eq!(n_matches, 2, "matches is wrong: {:#?}", matches);

        let applications = commute_plus.apply(&mut egraph, &matches);
        egraph.rebuild();
        assert_eq!(applications.len(), 2);

        let actual_substs: Vec<Subst> = matches.iter().flat_map(|m| m.substs.clone()).collect();

        println!("Here are the substs!");
        for m in &actual_substs {
            println!("substs: {:?}", m);
        }

        egraph.dot().to_dot("target/simple-match.dot").unwrap();

        use crate::extract::{AstSize, Extractor};

        let ext = Extractor::new(&egraph, AstSize);
        let (_, best) = ext.find_best(plus_id);
        eprintln!("Best: {:#?}", best);
    }

    #[test]
    fn nonlinear_patterns() {
        crate::init_logger();
        let mut egraph = EGraph::default();
        egraph.add_expr(&"(f a a)".parse().unwrap());
        egraph.add_expr(&"(f a (g a))))".parse().unwrap());
        egraph.add_expr(&"(f a (g b))))".parse().unwrap());
        egraph.add_expr(&"(h (foo a b) 0 1)".parse().unwrap());
        egraph.add_expr(&"(h (foo a b) 1 0)".parse().unwrap());
        egraph.add_expr(&"(h (foo a b) 0 0)".parse().unwrap());
        egraph.rebuild();

        let n_matches = |s: &str| s.parse::<Pattern<S>>().unwrap().n_matches(&egraph);

        assert_eq!(n_matches("(f ?x ?y)"), 3);
        assert_eq!(n_matches("(f ?x ?x)"), 1);
        assert_eq!(n_matches("(f ?x (g ?y))))"), 2);
        assert_eq!(n_matches("(f ?x (g ?x))))"), 1);
        assert_eq!(n_matches("(h ?x 0 0)"), 1);
    }

    #[test]
    fn search_with_limit() {
        crate::init_logger();
        let init_expr = &"(+ 1 (+ 2 (+ 3 (+ 4 (+ 5 6)))))".parse().unwrap();
        let rules: Vec<Rewrite<_, ()>> = vec![
            rewrite!("comm"; "(+ ?x ?y)" => "(+ ?y ?x)"),
            rewrite!("assoc"; "(+ ?x (+ ?y ?z))" => "(+ (+ ?x ?y) ?z)"),
        ];
        let runner = Runner::default().with_expr(init_expr).run(&rules);
        let egraph = &runner.egraph;

        let len = |m: &Vec<SearchMatches<S>>| -> usize { m.iter().map(|m| m.substs.len()).sum() };

        let pat = &"(+ ?x (+ ?y ?z))".parse::<Pattern<S>>().unwrap();
        let m = pat.search(egraph);
        let match_size = 2100;
        assert_eq!(len(&m), match_size);

        for limit in [1, 10, 100, 1000, 10000] {
            let m = pat.search_with_limit(egraph, limit);
            assert_eq!(len(&m), usize::min(limit, match_size));
        }

        let id = egraph.lookup_expr(init_expr).unwrap();
        let m = pat.search_eclass(egraph, id).unwrap();
        let match_size = 540;
        assert_eq!(m.substs.len(), match_size);

        for limit in [1, 10, 100, 1000] {
            let m1 = pat.search_eclass_with_limit(egraph, id, limit).unwrap();
            assert_eq!(m1.substs.len(), usize::min(limit, match_size));
        }
    }
}
