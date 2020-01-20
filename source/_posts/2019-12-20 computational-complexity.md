---
title: 计算复杂性-笔记
date: 2019-12-20 20:00:49
updated: 2019-12-24
categories: 砂时计
tags:
  - 计算理论
---

<br/>

# 前言

这是**计算复杂性**的课程笔记，推荐读物是: 

  - [Luca Trevisan. Lecture Notes on Computational Complexity](/downloads/pdf/Lecture&#32;Notes&#32;on&#32;Computational&#32;Complexity&#32;-&#32;Luca&#32;Trevisan.pdf)
  - Sanjeev Arora & Boaz Barak. Computational Complexity: A Modern Approach.
  - Chritos H. Papadimitriou. Computational Complexity.
  - Elaine Rich. Automata, Computability and Complexity: Theory and Application.
  - Oded GoldReich. Computational Complexity: A Conceptual Perspective


# Prelimilary

History briefly on thinking of computation:

```
Calculus                             Newton-Leibniz        1687
  (what is infinitesimal?)
1817: Epsilon-delta definition       Cauchy, Bolzano       1817
  Rigorous def. of infinitesimal
Set theory                           Cantor                1880s
  (what is infinite?)
Russell paradox                      Russell               1901
  (what is a set?)
ZFC axioms                           Zermelo–Fraenkel      1922
Hilbert program                      Hilbert               1920s
  Soundness/consistency
  Completeness
Gödel's incompleteness theorems      Gödel                 1931
  Gödel's first incompleteness theorem
  Gödel's second incompleteness theorem
What is computation?
  Recursive functions                Gödel                 1931  
  Turing machines                    Turing                1936
  Lambda calculus                    Church                1930s
```

Compuation models:

```
1936   Turing                 Turing machine
1936   Church                 Lambda caculus
1933   Gödel                  Recursive functions
1040s  Ullman, von Neumann    Cellular automata
1960s  Lambek, Minsky         Radom access machine
1985   Deutsch                Quantum Turing Machine
```


# TM & P

Time constructible functions: A function T: N -> N is time-constructible if T(n) >= n and there is a TM M that computes [T(|x|)], here [] is the syntax of floor()

Given f: {0,1}* -> {0,1} and time-constructible function T: N -> N,

  - if f is computable in time T(n) by a TM using alphabet Γ, then it is computable in time `4log|Γ|T(n)` by a TM M' using alphabet {0,1,□,▷}
  - if f is computable in time T(n) by a TM using k-tapes, then it is computable in time `5kT²(n)` by a single tape TM M'
  - if f is computable in time T(n) by a bidirectional TM M, then it is compuatable in time `4T(n)` by a standard TM M'
  - if f is computable in time T(n) by a standard TM M, then it is compuatable in time `CT(n)logT(n)` by an **oblivious TM M'** (oblivious: the location of head at the i-th step is a function only of |x| and i)

Indexed TMs: map all TMs to Σ*

  - every string in Σ* represents **a certain** TM
  - every TM is represented by **infinitely many** strings (functionally equal up to redundant states etc.)

UTM: there exists a TM U such that for ∀i, w ∈ Σ*, U(i, w) = M_i(w), moreover, if M_i(w) halts within T steps, then U(i, w) halts within CTlogT steps where C is a constant only depends on M_i's alphabet size

Computability:

  - The uncomputable function UC: UC(w) := if M_w(w) = 1 then 0 else 1 (constructed using diagonal method)
  - The HALT problem: HALT(i, w) := if M_i(w) halts then 1 else 0
  - Reduction from HALT to UC: Muc(w) = if U(w,w) = M_w(w) = 1 then 0 else 1

Functional class DTIME: given T: N -> N, a language is in DTIME(T(n)) iff there is a TM decides L that runs in time O(T(n))

Class P: `P = ∪[c>=1]DTIME(n^c)`

Problems in `P`:

 - Graph connectivity
 - {<i,x,y>: i-th bit of xy is 1}
 - Bipartite
 - Tree


# NP and NP-completeness

Class NP: a language L is in NP iff there exists a polynomial p: N -> N and a poly-time TM M such that for ∀x ∈ Σ*, x ∈ L iff ∃u∈Σ^(p(|x|)) st. M(x, u) = 1 (hence M is called the verifier of L, and u is the ceritificate for x is in L)

Problems in `NP`:

  - Maximum independt set (NP-complete)
  - Traveling salesman (NP-complete)
  - Subset sum (NP-complete)
  - 0/1 integer programming (NP-complete)
  - Graph isomorphism (not known, maybe NP-medium)
  - Factoring (not known, maybe NP-medium)
  - Linear programming (even in P)
  - Composite number (even in P, another hand PRIMES is in coNP and even in P)
  - Graph connectivity (even in L)

Class EXP: `EXP = ∪[c>1]DTIME(2^n^c)`, notice here `c>1` has no equal

Time complexity hierarchy till now: `P ⊆ NP ⊆ EXP`

Functional class NTIME: given T: N -> N, a language is in DTIME(T(n)) iff there is a NTM decides L that runs in time O(T(n))

Class NP (alter-def): `NP = ∪[c>=1]NTIME(n^c)`

Karp/poly-time reduction: there is a poly-time computable f: Σ* -> Σ* that maps L to L' such that for ∀x∈Σ*, x∈L iff f(x)∈L', then L is karp-reducible to L', denoted by `L <=p L'` (intuitively, deciding L' is harder then deciding L)

NP-hardness: if ∀L'∈NP, L' <=p L, then L is NP-hard  
NP-completeness: if L is in NP and is NP-hard, then L is NP-complete  

P ?= NP: iff ∃L is NP-complete and L∈P

Problems in `NP-complete` and their hierarchy:

  - TMSAT = {<i,x,1^n,1^t> | ∃u∈Σ^n, st. M_i(<x, u>) = 1 within t steps}
  - SAT [Cook-Levin Theorem]: langauge of all satisfiable CNF formula
    - 3SAT [Cook-Levin Theorem]: langauge of all satisfiable 3CNF formula
      - exact3SAT
        - SUBSETSUM
      - INDSET: {<G,k>: ∃S⊆V(G) st. |S|>=k and ∀u,v∈S, (u,v)∉E(G)}
        - CLIQUE
        - VERTEXCOVER
          - MAXCUT
      - 3COL
    - dHAMPATH: set of all direct graphs that contains a Hamilton path
      - HAMPATH
        - TSP: traveling sales person
        - HAMCYCLE
    - INTEGERPROG
    - THEOREM
    - QUADEQ

Class coNP: `coNP = {L: ~L∈NP}`, where ~L means (Σ* - L)

Class coNP(alter-def): a language L is in coNP iff there exists a polynomial p: N -> N and a poly-time TM M such that for ∀x ∈ Σ*, x ∈ L iff ∀u∈Σ^(p(|x|)) st. M(x, u) = 1 （notice here is **∀** compared with **∃** in NP's alter-def)

Problems in `coNP`:

  - TAUTOLOGY: set of all boolean formula that is satisfied by any assigment (always true)
  - PRIMES

Padding technique: to prove `if P = NP, then EXP = NEXP`, for ∀L in NEXP constuct Lpad = {<x, 1^2^|x|^c>: x∈L}, then to prove Lpad in NP, thus by hypotheis we know Lpad is in P, by `P ⊆ EXP` we know Lpad is in EXP


# Diagonalization

Time hierarchy theorem: if f, g is time-constructible functions satisfying `f(n)logf(n)= o(g(n))`, then `DTIME(f(n)) ⊊ DTIME(g(n))` (intuitively given an **log times more** time, indeed more TMs would be included to the DTIME set)

```
Proof for TH: try to find a TM D in DTIME(g(n)) but not in DTIME(f(n))

construct the diagonal table:

    x\TM M0 M1 ...  Mx  ...
     0
     1
    ...        ...
     x             Mx(x)
    ...                 ...

construct a DTM D, D(x) simulates UTM(Mx, x), if UTM(Mx, x) halts within g(|x|) steps then D(x) = ~UTM(Mx, x), otherwise D(x) = 0
```

Collary: `P ⊊ EXP` due to `P ⊆ DTIME(2^n) ⊊ DTIME(2^n^2) ⊆ EXP`

Non-deterministic time hierarchy theorem: if f, g is time-constructible functions satisfying `f(n+1) = o(g(n))`, then `NTIME(f(n)) ⊊ NTIME(g(n))` (proof is hard, use lazy diagnalize)

Collary: `NP ⊊ NEXP` due to `NP ⊆ NTIME(2^n) ⊊ NTIME(2^n^2) ⊆ NEXP`

NP-medium [Ladner Theorem]: if P != NP, then there exists a L ∈ NP\P and L ∉ NP-complete. (ie. that L is SAT_H = {ψ01^(nH(n)) | ψ∈SAT, n=|ψ|}, H(n): the smallest number i < loglongn such that for ∀x∈Σ* with |x| < logn, Mi outputs SAT_H(x) within i|x|^i steps, if there is no such number i then H(n) = loglogn)

Oracle TM: a normal TM M or a TM set Ms (eg. a complexity class) with oracle access to certain language L or a language class Ls, denoted by `M^SAT` or `M^NP` or `P^HAMPATH` or `NP^EXP` alike, M or Ms can query whether a string is in L or in any language of Ls use just one step

Diagonal method could not help proving P != NP: thoerem, there exists oracle A,B such that P^A = NP^A while P^B != NP^B (intuitively, P ?= NP is not a relativizing result)


# Space complexity

Functional class SPACE: given s: N -> N, a language is in SPACE(s(n)) iff there is a TM decides L that at most O(s(n)) cells on working tape is ever visited  
Functional class NSPACE: given s: N -> N, a language is in NSPACE(s(n)) iff there is a NTM decides L that at most O(s(n)) cells on working tape is ever visited  

Space constructible functions: A function s: N -> N is space-constructible if there is a TM M that computes s(|x|) in O(S(|x|)) space given

Space complexity hierarchy till now: `DTIME(S(n)) ⊆ SPACE(S(n)) ⊆ NSPACE(S(n)) ⊆ DTIME(2^O(S(n)))` for any space-construtible S: N -> N

Configuration graph: given certian input x and TM M, the transition flow between configurations at runtime of M(x) forms a configuration graph (a DAG, pseudo-tree)  
*use this to prove `NSPACE(S(n)) ∈ DTIME(2^O(S(n)))`, hence time complexity and space complexity is tied up

Class PSPACE: `PSPACE = ∪[c>=1]SPACE(n^c)`  
Class NPSPACE: `NPSPACE = ∪[c>=1]NSPACE(n^c)`  
Class L: `L = ∪[c>=1]SPACE(logn)`  
Class NL: `NL = ∪[c>=1]NSPACE(logn)`  

Problems in these spacial classes:

  - 3SAT ∈ SPACE(n) ⊆ PSPACE
  - EVEN = {x | x has even number of 1's} ∈ L
  - MULT = {<1^m, 1^n,1^mn>} ∈ L
  - PATH = {<G, s, t> | there is a path from s to t in dircted graph G}∈ NL

Space hierarchy theorem: if f, g is time-constructible functions satisfying `f(n)= o(g(n))`, then `SPACE(f(n)) ⊊ SPACE(g(n))` (intuitively given an **slightly times more** time, indeed more TMs would be included to the SPACE set)

Collary: `L ⊊ PSPACE` due to `L ⊆ SPACE(n) ⊊ SPACE(n^2) ⊆ PSPACE`

PSPACE-hardness: if ∀L'∈PSPACE, L' <=p L, then L is PSPACE-hard  
PSPACE-completeness: if L is in PSPACE and is PSPACE-hard, then L is PSPACE-complete  

Problems in `PSPACE-complete`:

  - SPACETMSAT = {<M, W, 1^n>: M(w) = 1 run in space n}
  - TQBF = {ψ∈QBF: ψ∈TAUTOLOGY}, QBF is a qualified boolean formula (compared that SAT is unqualified), note that **TQBF is also NPSPACE-hard**

Savitch's theorem: for any space-constructible function S: N -> N with S(n) >= logn, NSPACE(S(n)) ⊆ SPACE(S²(n)), thus `PSPACE = NPSPACE`

Log-space reduction: there is a **implicitly** log-space computable f: Σ* -> Σ* that maps L to L' such that for ∀x∈Σ*, x∈L iff f(x)∈L', then L is logspace-reducible to L', denoted by `L <=l L'`  
\*implicitly log-space computable: a function f: Σ* -> Σ* that is **poly-time bounded** (there is some constant c such that |f(x)| <= |x|^c for ∀x∈Σ*), Lf := {<x, i>: f(x)[i] = 1, aka. the i-th bit of f(x) is 1} and Lf' := {<x, i>: i <= |f(x)|} are in complexity class L

NL-hardness: if ∀L'∈NL, L' <=l L, then L is NL-hard  
NL-completeness: if L is in NL and is NL-hard, then L is NL-complete  

Problems in `NL-complete`: PATH

Class NL (alter-def): a language L is in NL iff there exists a polynomial p: N -> N and a poly-time TM M with a **read-once** tape for the ceritificate, such that for ∀x ∈ Σ*, x ∈ L iff ∃u∈Σ^(p(|x|)) st. M(x, u) = 1

Thoerem: `NL = coNL`  
Corollary: `NPSPACE(S(n)) = coNPSPACE(S(n))`, for any space-constructible S(n) > logn (using padding technique) (eg. PSPACE = NPSPACE)

Complexity hierarchy till now: `L ⊆ NL ⊆ P ⊆ NP ⊆ PSPACE = NPSPACE ⊆ EXP`


# Polynomial Hierarchy

Class Σp[2]: a language L is in Σp[2] iff there exists a polynomial p and a poly-time TM M such that for ∀x ∈ Σ*, x ∈ L iff ∃u∈Σ^(p(|x|))∀v∈Σ^(p(|x|)) st. M(x, u, v) = 1  
*obviously it is an extension of NP class, and can be generalized to Σp[k] where k is a constant integer  

Problems in `Σp[2]`:

  - EXACT INDSET: {<G, k>: G has an independent set sized k}
  - MIN-EQ-DNF: {<φ, k>: ∃DNF ψ of size <= k that is equivalent to DNF φ}

Class Σp[k]: a language L is in Σp[k] iff there exists a polynomial p and a poly-time TM M such that for ∀x ∈ Σ*, x ∈ L iff ∃u1∀u2∃u3..Quk st. M(x, u1, u2, ..., uk) = 1  
Class Πp[k]: `Πp[k] = coΣp[k]`  

Class PH: `PH = ∪[i>=1]Σp[k]`  
*NP = Σp[1], coNP = Πp[1], PH ⊆ PSPACE  

Problems in `Σp[k]-complete`: ΣkSAT = {φ | ∃u1∀u2∃u3..Quk φ(u1, u2, ..., uk) = 1}  
Class Σp[k] (alter-def): `Σp[k] = NP^Σ(k-1)SAT`, eg. `Σp[2] = NP^SAT = NP^NP`

Theorems on `PH`:

  - for ∀i>=1, if Σp[i] = coΣp[i] = Πp[i], then PH = Σp[i] (PH collapse to the i-th layer)
  - if P = NP, then PH = P (PH collapse to P, ie. 0-th layer)

Conjecture: `PH does not collapse`, `PH != PSPACE`

Alternative TM: every state of TM except q_start/q_halt is marked with ∀ or ∃  
*for given an ATM M and input x, draw the configuration graph of M(x), and from configurations whose state is q_accept, recursively mark each configuration node Ccur with 'ACCEPT' if (Ccur is in ∃ state and has at least one child been marked 'ACCEPT') or (Ccur is in ∀ state and all its childeren been marked 'ACCEPT'), finally set M(x) = 1 iff Cstart is marked 'ACCEPT'  

Functional class ATIME: given T: N -> N, a language is in ATIME(T(n)) iff there is a ATM decides L that runs in time O(T(n))
Class AP: `A = ∪[c>=1]ATIME(n^c)`, and we just have `AP = PSPACE`  
Functional class ΣkTIME/ΠkTIME: given T: N -> N, exists an ATM M whose initial state marked ∃/∀ decides L with at most k-1 times label change that runs in time O(T(n))
Class Σp[k] (alter-def): `Σp[k] = ∪[i>=1]ΣkTIME[n^c]`  
Class Πp[k] (alter-def): `Πp[k] = ∪[i>=1]ΠkTIME[n^c]`  

Functional class TISP: given S, T: N -> N, a language is in TISP(T(n), S(n)) iff there is a TM decides x ∈ L that runs in time O(T(|x|)) and space O(S(|x|))  
Time/space tradeoff: theorem, `SAT ∉ TISP(n^1.1, n^0.1)`  


# Boolean circuits

Functional class SIZE: given T: N -> N, a language is in SIZE(T(n)) iff there is a T(n)-size circuit family {Cn} decides L

Problems in `SIZE(n)`:

  - {1^n: n∈N}
  - {(m,n,m+n): m,n∈N}

Class P/poly: `P/poly = ∪[c>=1]SIZE(c)`, clearly we have `P ⊆ P/poly`  
*inconsistency of P/poly: `UHALT = {1^n: <M,x> encoding as unary string for which M(x) halts} ∈ P/poly`, even if UHALT is turing-uncomputable

Problem of CKT-SAT:

  - CKT-SAT = { string represention of all one bit fanout circuits which has at least one satisfying input }
  - CKT-SAT is NP-complete
  - CKT-SAT <=p 3SAT

Class P-uniform: a circuit family {Cn} is P-uniform iff there is a poly-time TM that on input 1^k outputs the string represented circuit of Ck  
*theorem, a language L is computed by a P-uniform circuit family iff L∈P  

Class logspace-uniform: a circuit family {Cn} is logspace-uniform there is a **implicit logspace computable function f** that maps 1^k to the string represented circuit of Ck, where that **f** requires these functions is logspace computable:

  - SIZE(k): size (vertex number) of circuit Ck
  - TYPE(k, i): type the i-th vertex in circuit Ck, one of {AND, OR, NOT, -(for i/o)}
  - EDGE(k,i,j): whether there is a directed edge in Cn from vertex i to j

*theorem, a language L has logspace-uniform circuits of polynomial size iff L∈P  

Adviced TM: alike Oracle TM, however allowed access is not a oracle function but α(|x|) bit of advice, time of run denoted `DTIME(T(n))/α(n)`  
*UHALT∈DTIME(n)/1, just need on bit to show x = 1^k is in UHALT  

Class P/poly (alter-def): `P/poly = ∪[c,d>=1]DTIME(T(c))/α(d)`, just set the advice to be the string represented circuit of Ck

NP vs P/poly [Karp-Lipton Theorem]: if NP ⊆ P/poly, then PH = Σp[2]  
EXP vs P/poly [Meyer's Theorem]: if EXP ⊆ P/poly, then EXP = Σp[2]  
*chain them up: **if P = NP**, then (P = NP) => (P = Σp[2]) => (EXP != Σp[2]) => (EXP ⊊ P/poly)

Hard functions [Shannon49]: for ∀n>1, ∃f: Σ* -> Σ that cannot be computed by a circuit of/under size 2^n/(10*n) (proof is EZ, just compare numbers of funtions with circuits)  
Non-uniform hierarchy theorem: for ∀T, T': N -> N with 2^n/n > T'(n) > T(n), SIZE(T(n)) ⊊ SIZE(T'(n))  

Functional class NC^d: a language is in NC^d iff exists a circuit family {Cn} with poly(n) size and O(log^d(n)) depth decides L
Class NC: `NC = ∪[i>=1]NC^i`, this class is important for indicating problems that has **sufficient parallel algorithms**  
Functional class AC^d: alike NC^d but gates are allowed to have unbounded fan-in other than regularly 2 (thus it folds up some depth)  
Class AC: `AC = ∪[i>=1]AC^i`  
Problems in `NC`: PARITY = {x: x has odd number of 1s} is in `NC^1`  
Circuit lower bound: `PARITY ∉ AC^0`, known as 'complexity theory's waterloo'

P-completeness: if L is in P and ∀L'∈P, L' <=l L  
Problems in `P-complete`: CIRCUIT-EVAL = {<C, x>: C is a |x|-input single-output circuit st. C(x)=1}


# Randomized computation

Probabilistic TM: a TM with two transition functions, it tosses a (fair) coin for choosing to use which one at each step, thus ouput of the PTM might vary even with the same input (de-randomize: once its coin results is embbed as a part of inputs, a PTM turns to be a DTM)

Functional class BPTIME: given a T: N -> N, a language is in BPTIME(T(n)) iff a PTM decides L with probability p>=2/3 (ie. Pr[M(x) = L(x)] >= 2/3) that runs in time O(T(n))  
*the constant 2/3 could just be any value strictly bigger than 1/2, they all could be error-reduced to nearly 1 using chernoff inequality by simply repeat the PTM enough times and set final answer to the most freqent answer
Class BPP: `BPP = ∪[c>=1]BPTIME(n^c)`, we have `BPP ⊆ EXP`, `BPP ⊆ P/poly`, `BPP ⊆ Σp[2] ∩ Πp[2]`
Class BPP (alter-def): a language L is in BPP iff there exists a polynomial p: N -> N and a poly-time TM M such that for ∀x ∈ Σ*, Pr(r∈Σ^p(|x|))[M(x,r) = L(x)] >= 2/3

Problems in `BPP`:

  - PRIME: Primality
  - KTHELEM: Find k-th element/median
  - ZEROP: Polynomial identity/zero
  - Bipartite graph perfect matching

Functional class RTIME: alike BPTIME, but require one-side error, ie. (x ∈ L iff Pr[M(x) = L(x)] >= 2/3) and (x ∉ L iff Pr[M(x) = 0] = 1)  
Class RP: `RP = ∪[c>=1]RTIME(n^c)`  
*de facto, RP ⊆ NP and coRP ⊆ coNP, ZEROP ∈ RP

Functional class ZTIME: alike BPTIME, but require zero-side error, ie. Pr[M(x) = L(x)] = 1, but runs in **expected running time** O(T(n))  
*expected running time: given PTM M and input x, considering the full-space of random bits r used in compute M(x) is |Σ|^p(|x|), define 'expected running time' as the expectation of steps for running M(x) on the random variable r  
Class ZPP: `ZPP = ∪[c>=1]ZTIME(n^c)`  
*de facto, ZPP = RP ∩ coRP, ZTIME(T(n)) ⊆ BPTIME(T(n))

Fair and unfair coin: could simulate each other, design a random generator for given distribtion function is a trick intellgent work :(

Randomized reduction: there is a PTM M for ∀x∈Σ*, st. Pr[L'(M(x)) = L(x)] >= 2/3, then L is random-reducible to L', denoted by `L <=r L'`
*note that '<=r' is not transitive
Class BP·NP: `BP·NP = {L: L <=r 3SAT}`

Class BPL: a language is in BPL iff exists a log-space PTM M st. Pr[M(x) = L(x)] >= 2/3  
Class RL: alike BPL, but one-side error (ie. no tolerance for x ∉ L)
*de facto, UPATH ∈ RL (UPATH is even in L)


# Interactive Proofs

Functional class IP[k]: for an interger/function k>=1, a language L is in IP[k] iff there exists a PTM V that can have a k-round interaction with a function P: Σ* -> Σ* st. (x ∈ L iff ∃P Pr[out_v<V, P>(x) = 1] >= 2/3) and (x ∉ L iff ∀P Pr[out_v<V, P>(x) = 1] <= 1/3)  
*interaction between V (as f) and P (as g) are a sequence of messages like: `f(x) = a1, g(x, a1) = a2, f(x, a1, a2) = a3, ...`
Class dIP: `dIP = ∪[c>=1]IP[n^c], but TM for V is a DTM rather than PTM `  
Class IP: `IP = ∪[c>=1]IP[n^c]`

Facts of IP:

  - whether P is DTM or PTM doesn't change the defined class
  - IP ⊆ PSPACE
  - replacing the completeness paramter 2/3 to 1 doesn't change the defined class
  - replacing the soundness paramter 1/3 to 0 will change the defined class to NP
  - the paramters could be amplified to nearly 1 and 0 by parallel repeatition, so doesn't increase the number of interact rounds

Problems in `IP`:

  - GNI: graph nonisomorphism (even in coNP, AM)
  - QNR: quadratic nonresiduosity (even in coNP)

Functional class ZKP: too long, didn't read :(...  
Problems in `ZKP`: GI, graph isomorphism (even in NP, maybe NP-medium)  
*if GI is NP-complete, then Σp[2] = Πp[2] (thus PH collapse to Σp[2])

Functional class AM[k]: alike IP[k], but restrict V can only send random bits to P, and V cannot use any other random bits other than those has sent to P
Class AM: `AM = ∪[c>=k]AM[k]`, de facto `AM = AM[2] = AM[k], k>=2`, and `AM = BP·NP`

Problems in `AM`: GNI

Class #P and #P-complete problem #SAT_D: #SAT_D = {<φ, k>: φ is a 3CNF with exactly k satisfying assignments}, we have `#SAT_D ∈ IP`


# Finale: complexity zoo

See here, the universe of [Complexity Zoo](https://complexityzoo.uwaterloo.ca/).

Summary for those classes that we've covered:

```
                                                     PCP(logn,1)                            PCP(polyn,1)
               NC^2                                     ||                                     ||
NC^1 ⊆ L ⊆ NL ⊆ NC ⊆ P ⊆                         ⊆ NP ⊆ PH ⊆ PSPACE = AP = IP ⊆ EXP ⊆ NEXP = MIP
          |coNL|       |AL|          |BP·P/coBPP|     |dIP|   |coNPSPACE/NPSPACE|    |APSPACE|
                            ZPP ⊆ RP ⊆ BPP ⊆ P/poly
                                      IP[O(1)] ⊆ AM = BP·NP

[the proved]
NL = coNL, NPSPACE = coNPSPACE
P ⊆ NP ∩ coNP, P ⊆ P/poly
L ⊊ PSPACE, P ⊊ EXP, NP ⊊ NEXP     ; hierarchy theorem

NC^1 ⊆ L ⊆ NL ⊆ NC^2
NC ⊆ P

BPP ⊆ EXP, BPP ⊆ P/poly, BPP ⊆ Σp[2] ∩ Πp[2]
RP ⊆ BPP, coRP ⊆ BPP
ZPP = RP ∩ coRP

#P ⊆ IP
IP[k] ⊆ AM[k+2], IP[O(1)] ⊆ AM
AM = AM[2] = AM[k] = BP·NP

[the guesses]
P = NC = BPP = u-P/poly
NP = BP·NP
```
