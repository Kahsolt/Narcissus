---
title: 计算机程序形式语义-笔记
date: 2019-12-20 20:02:49
updated: 2019-12-24
categories: 砂时计
tags:
  - 计算模型
---

<br/>

# 前言

这是**计算机程序形式语义**的课程笔记，推荐读物是: 

  - John C. Reynolds. Theories of Programming Languages
  - [Lecture Notes on the Lambda Calculus - Peter Selinger](/downloads/pdf/Lecture&#32;Notes&#32;on&#32;the&#32;Lambda&#32;Calculus&#32;-&#32;Peter&#32;Selinger.pdf)
  - [Separation Logic - copenhagen08](/downloads/pdf/Separation&#32;Logic&#32;-&#32;copenhagen08.pdf)


# Lambda calculus

Invented in 1930s, by Alonzo Church and Stephen Cole Kleene

Alan Turing, 1937: Turing machines equal λ-calculus in expressiveness

## Pure LC

### Syntax

Syntax for pure λ-calculus terms/expressions:

```
(Terms) M, N ::= x | λx. M | M N
    x         lexical atomic literal stands for a variable
    λx. M     lambda abstraction, i.e. anonymous function
    M N       lambda application, i.e. function call
and brackets `()` to adjust priority
```

Conventions to omit brackets:

  - body of lambda abstraction extends as far to right as possible: `λx. M N` means `λx. (M N)`,  not `(λx. M) N`
  - lambda application is left-associative: `M N P` means `(M N) P`,  not `M (N P)`

Often we also add symbols for extra operations and native data types support, so that we could also write these and take them as valid terms:

```
λz. (x+2*y+z)
(λx. (x+1)) 3
(λz. (x+2*y+z)) (λx. (x+1))
```

Higher-order function is natural:

```
  (λf. λx. f (f x)) (λy. y+1) 5
= (λx. (λy. y+1) ((λy. y+1) x)) 5
= (λx. (λy. y+1) (x+1)) 5
= (λx. (x+1)+1) 5
= 5+1+1
```

Free and bound variables: in `(λz. λx. x + y - z) (z+1)`, x is bounded, y is free, z is both bounded and free; recursively define a function `fv(M)` to get the free variables:

```
recall that: M, N ::= x | λx. M | M N
thus:
    fv(x) = {x}
    fv(λx. M) = fv(M) \ {x}
    fv(M N) = fv(M) ∪ fv(N)
```

### Semantics

Reduction rules:

```
(λx. M) N -> M[N/x]                ; most basic, namely β-reduction, replace x by N in M
if M -> M', then M N -> M' N       ; simplify compounded function
if N -> N', then M N -> M N'       ; simplify argument
if M -> M', then λx. M -> λx. M'   ; simplify function body
```

But consciously avoid name capture, e.g:

```
  (λx. x-y)[x/y]    ; could NOT directly reduce to `λx. x - x`
= (λz. z-y)[x/y]    ; so, always rename bound variables before substitution
= λz. z-x

  (λx. f (f x))[(λy. y+x)/f] 
= (λz. f (f z))[(λy. y+x)/f]     ; always rename
= λz. (λy. y+x) ((λy. y+x) z)
```

Normal form:

```
β-redex: a term of the form (λx.M) N
β-normal form: a term containing no β-redex, thus could no more simplify by β-reduction

Church-Rosser property: terms can be evaluated in any order, but final result (if there is one) is uniquely determined - the β-normal form. (You can visualize it as a diamond graph)
*collary: with α-equivalence(differs only in variable names), every term has at most one normal form
```

Non-terminating reduction, terms which have no normal forms:

```
  (λx. x x) (λx. x x)
= (λx. x x) (λx. x x) 
= ...

  (λx. x x y) (λx. x x y) 
= (λx. x x y) (λx. x x y) y 
= ...

  (λx. f (x x)) (λx. f (x x)) 
= f ((λx. f (x x)) (λx. f (x x))) 
= ...
```

Term may have both terminating and non-terminating reduction sequences:

```
  (λu. λv. v) ((λx. x x)(λx. x x))    ; normal-order, simplify function first
= λv. v

  (λu. λv. v) ((λx. x x)(λx. x x))    ; applicative-order, simplify argument first
= (λu. λv. v) ((λx. x x)(λx. x x)) 
= ...
```

Reduction strategy:

  - normal-order: simplify function first, will eventually find a normal form if exists
  - applicative-order: simplify argument first, in most cases could avoid evaluating reduplicated terms

Evaluation is just like a half-redcution, it:

  - only evaluate closed terms (i.e. no free variables)
  - does not reduce under lambda, (ie. not simplify function body, **might** avoid ill cases that does not terminate), therefore it stops once meet a **canonical form** (ie. a lambda abstraction, a function)

Normal-order evaluation rules (corresponding to normal-order reduction):

```
[big step]
λx. M => λx. M                       ; not simplify function body
if (M => λx. M') and (M'[N/x] => P)  ; funtion simplifiable
    then M N => P                    ; β-reduction for application term

[small step]
(λx. M) N -> M[N/x]                  ; β-reduction
if M -> M', then M N -> M' N         ; simplify compounded function
```

Eager evaluation rules (corresponding to applicative-order reduction):

```
[big step]
λx. M =>E λx. M                       ; still, not simplify function body
if (M =>E λx. M') and (N =>E N') and (M'[N'/x] =>E P)  ; both funtion and argument simplifiable
     then M N =>E P                   ; β-reduction for application term

[small step]
(λx. M) (λy. N) -> M[(λy. N)/x]           ; β-reduction
if M -> M', then M N -> M' N              ; simplify compounded function
if N -> N', then (λx. M) N -> (λx. M) N'  ; simplify argument
```

### Programming

这是部分基础LC的python代码实现: [lambda_calculus.py](/downloads/src/lambda_calculus.py)

```
[Logical]
True  ::= λx. λy. x
False ::= λx. λy. y
not   ::= λb. b False True
not'  ::= λb. λx. λy. b y x
and   ::= λb. λb'. b b' False
or    ::= λb. λb'. b True b'
if b then M else N ::= b M N 

[Number]
0 ::= λf. λx. x   
1 ::= λf. λx. f x
2 ::= λf. λx. f (f x)
...
n ::= λf. λx. f^n x

succ   ::= λn. λf. λx. f (n f x) 
succ'  ::= λn. λf. λx. n f (f x)
iszero ::= λn. λx. λy. n (λz. y) x
add    ::= λn. λm. λf. λx. n f (m f x)
mult   ::= λn. λm. λf. λx. (n (m f)) x

[Pair/Tuple]
(M, N) ::= λf. f M N
π0     ::= λp. p (λx. λy. x)
π1     ::= λp. p (λx. λy. y)

[Fixpoint combinator]
Turing’s fixpoint combinator Θ:
  A ::= λx. λy. y (x x y)
  Θ = A A
Church’s fixpoint combinator Y:
  Y ::= λf. (λx. f (x x)) (λx. f (x x))

*in λ-calculus, every term has a fixpoint
fixpoint combinator is a higher-order function h satisfying:
  forall f, st. h f = f (h f), i.e. (h f) is a fixpoint of f
```

## Simply-typed LC

Idea: try kicking out those non-terminating terms by adding a type system

  - type check would catch simple mistakes early in compile time
  - type-safety: well-typed programs never stuck into meaningless state (out of semantic specification)
  - typed programs are easier to analyze and optimize
  - but impose constraints on programmers, and some valid programs **might** be rejected

### Syntax & Semantics

Syntax for simply-typed λ-calculus (STLC):

```
(Types) σ, τ ::= T | σ→τ     ; T is some base type, eg. int/bool
(Terms) M, N ::= x | λx:τ. M | M N
```

Reduction rules are much like those in pure LC:

```
(λx:τ. M) N -> M[N/x]
if M -> M', then M N -> M' N
if N -> N', then M N -> M N'
if M -> M', then λx:τ. M -> λx:τ. M'

*strong normalization theorem: well-typed terms in STLC always terminate
```

### Type judgment

```
Γ |- M:τ            ; type assertion, M is of type τ in contex Γ
Γ ::= ∙ | Γ, x:τ    ; typing contex, ∙ means empty, x is free variable in M

[typing rules]
Γ, x:τ |- x:τ                 ; var
if (Γ |- M:σ→τ) and (Γ |- N:σ)
    then Γ |- M N:τ           ; app
if (Γ, x:σ |- M:τ)
    then Γ |- (λx:σ. M):σ→τ  ; abs
```

Soundness and Completeness:

  - Soundness: never accepts a program that can go wrong
    - no false negatives 
    - the language is type-safe
  - Completeness: never rejects a program that can’t go wrong 
    - no false positives 
    - however, for any Turing-complete PL, the set of programs that may go wrong is undecidable

*However, type system cannot be both sound and complete: in practice, we choose soundness, then try to reduce false positives 

Therefore STLC is sound but incomplete:

  - Soundness: type-safety theorem
    - preservation (subject reduction): well-typed terms reduce only to well-typed terms of the **same** type
    - progress: a well-typed term is either a Value (defined in semantics) or can be reduced (ie. **eventually is a Value**)
  - Incompleteness: it rejects some valid term, eg. `(λx. (x (λy. y)) (x 3)) (λz. z)`, since it could not be type-judged, nevertheless it is reducible



### Extensions & Curry-Howard Isomorphism

Syntax and semantics:

```
(Types) σ, τ ::= T | σ→τ | σ×τ | σ+τ
(Terms) M, N ::= x | λx:τ. M | M N | <M, N> | proj1 M | proj2 M | left M | right M | case M do M1 M2 | fix M
(Values) V :: <M, N> | proj1 M | proj2 M | left M | right M

*note: by adding 'fix', strong normalization thereom breaks
```

Reduction rules:

```
[prod-type] & [sum-type]
...

[fix]
fix λx:τ. M -> M[fix λx:τ. M/x]   ; one recursive call
if M -> M', then fix M -> fix M'  ; simplify function body
```

Curry-Howard Isomorphism:

 - Propositions are Types
 - Proofs are Programs

```
(Prop)   p, q ::= B | p⇒q | p∧q | p∨q
(Types)  σ, τ ::= T | σ→τ | σ×τ | σ+τ

thus, several well-typed closed term maps to the same one type, then that type identically maps to a formula provable in propositional logic
```

Constructive logic: no `law of the excluded middle`

```
in classical logic there should be 'Γ |- p∨(p⇒q)'
but in STLC, no closed term has type 'ρ+(ρ→σ)'

btw, due to support of 'fix', the "logic" behind STLC is inconsistent
since type of 'fix λx:τ. x' is arbitrary 'τ', which means anything is provable
```


# Operational semantics

Operational semantics defines program executions as a sequence of steps, formulated as transitions of an abstract machine

Syntax of a simple imperative language:

```
(IntExp) e ::= n                  ; numerals, n denotes syntax, [n] denotes semantics
             | x                  ; variable names
             | e + e | e - e
(BoolExp) b ::= true | false
              | e = e | e < e | e > e
              | not b | b and b | b or b
(Comm) c ::= skip
           | x := e
           | c ; c
           | if b then c else c
           | while b do c

(States) σ ∈ Var -> Values        ； Var is x, Values is n
*eg.  σ1 = {(x, 2), (y, 3), (a, 10)} writes as {x~>2, y~>3, a~>10}
      σ1{y~>7} = {x~>2, y~>7, a~>10}   ; value update

(Configuration) S ::= (e, σ) | (b, σ) | (c, σ)
```

## Small-step

small-step semantics describes each single step of the execution

### Structural operational semantics (SOS):

```
[expression evaluation - int]
if (e1, σ) -> (e1', σ), then (e1 + e2, σ) -> (e1' + e2, σ)  ; evalue from left to right
if (e2, σ) -> (e2', σ), then (n + e2, σ) -> (n + e2', σ)
if [n1] [+] [n2] = [n], then (n1 + n2, σ) -> (n, σ)         ; substrcation is alike

[variable read]
if σ(x) = [n], then (x, σ) -> (n, σ)

[expression evaluation - boolean]
if (e1, σ) -> (e1', σ), then (e1 = e2, σ) -> (e1' = e2, σ)  ; evalue from left to right
if (e2, σ) -> (e2', σ), then (n = e2, σ) -> (n = e2', σ)
if [n1] [=] [n2] = [n], then (n1 = n2, σ) -> (true, σ)
if not ([n1] [=] [n2] = [n]), then (n1 = n2, σ) -> (false, σ)  ; other comparations are alike

if (b, σ) -> (b', σ), then (not b, σ) -> (not b', σ)
(not true, σ) -> (false, σ)
(not false, σ) -> (true, σ)

if (b1, σ) -> (b1', σ), then (b1 and b2, σ) -> (b1' and b2, σ) ; evalue from left to right
(false and b2, σ) -> (false, σ)                                ; short-circuit feature
(true and b2, σ) -> (b2, σ)

[statement]
(skip, σ) -> σ

if (e, σ) -> (e', σ), then (x := e, σ) -> (x := e', σ)
(x := n, σ) -> σ{x~->[n]}

if (c0, σ) -> (c0', σ'), then (c0 ; c1, σ) -> (c0' ; c1, σ')
if (c0, σ) -> σ', then (c0 ; c1, σ) -> (c1, σ')

if (b, σ) -> (b', σ), then (if b then c0 else c1, σ) -> (if b' then c0 else c1, σ)
(if true then c0 else c1, σ) -> (c0, σ)
(if false then c0 else c1, σ) -> (c1, σ)

(while b do c, σ) -> (if b then (c; while b do c) else skip, σ) ; just expand once
```

Some facts about '->':

  - cound be extended to multiple steps '->*'
  - Determinism: if (c, σ) -> (c', σ') and (c, σ) -> (c'', σ''), then (c', σ') = (c'', σ'') (i.e. rule to apply on each step is unambiguously unique)
  - Confluence: think of a diamond graph
  - Normalization: transition relations on (e, σ) and (b, σ) are normalizing, but NOT on (c, σ)
    - normal form for expressions: (n, σ) for all numeral n
    - normal form for booleans: (true, σ) and (false, σ)

Once we defined '->*', we could fold up evaluation sequence:

```
[variation-1]
if [[e]](intexpr)σ = n, then (x := e, σ) -> σ{x~>n}
*here '[[e]](intexpr)σ = n' iff '(e, σ) ->* (n, σ) and n = [n]'

if [[b]](boolexpr) = true, then (if b then c0 else c1, σ) -> (c0, σ)
if [[b]](boolexpr) = false, then (if b then c0 else c1, σ) -> (c1, σ)

if [[b]](boolexpr) = true, then (while b do c, σ) -> (c; while b do c, σ)
if [[b]](boolexpr) = false, then (while b do c, σ) -> σ

[variation-2]
if [[e]](intexpr)σ = n, then (x := e, σ) -> (skip, σ{x~>n})   ; this is an alternative for above one, if we want to keep configurations always tuples
; and other rules whose configuraion is a single 'σ' are also modified to '(skip, σ)'
```

We then extend the [variation-2] with abortion, local-var, dynamic-alloc:

```
[abortion]
(IntExp) e ::= ... | e1 / e2

if n2 != 0 and [n1] [/] [n2] = [n], the (n1 / n2, σ) -> (n, σ)
(n1 / 0, σ) -> abort

if [[e]](intexpr)σ = n, then (x := e, σ) -> (skip, σ{x~>n})
if [[e]](intexpr)σ = -, then (x := e, σ) -> abort
*here '[[e]](intexpr)σ = -' iff '(e, σ) ->* abort', thus we could not assign
; also cascadingly modify other rules

[local-var]
(Comm) c :: = ... | newvar x := e in c

if [[e]](intexpr)σ = n and (c, σ{x~>n}) ->(c', σ') and σ'(x) = [n'],
    then (newvar x := e in c, σ) -> (newvar x := n' in c', σ'{x~>σ(x)})
(newvar x := e in skip, σ) -> (skip, σ)
; could also add abortion

[dynamic-alloc]
(State) σ ∈ (s, h)
(Store) s ∈ Var -> Values
(Heap)  h ∈ Loc ->fin Values
(Value) v ∈ Int ∪ Bool ∪ Loc
(Comm) c :: = ... | x := alloc(e)  ; allocation, x is a Loc
                  | y := [x]       ; lookup, y is a Var
                  | [x] := e       ; mutation
                  | free(x)        ; deallocation
(Configuration) S ::= (c, (s, h))

if l ∉ dom(h) and [[e]](intexpr)s = n, then (x := alloc(e), (s, h)) -> (skip, (s{x~>l}, h{l~>n}))
if s(x) = l and l ∈ dom(h), then (free(x), (s, h)) -> (skip, (s, h\{l}))
if s(x) = l and h(l) = n, then (y := [x], (s, h)) -> (skip, s{y~>n}, h)
if s(x) = l and l ∈ dom(h) and [[e]](intexpr)s = n, then ([x] := e, (s, h)) -> (skip, s, h{l~>n})
```

### Contextual semantics (aka. reduction semantics) is more systematical:

```
if (r, σ) -> (e', σ), then (E[r], σ) -> (E[e'], σ)
  r ::= x | n + n | n - n | ...                    ; redex
  E ::= [] | E + e | E - e | n + E | n - E | ...   ; evaluation context (big ε in handwriting)

here is overall definition for redex and context of our simple imperative language:
(Redex) r ::= x
            | n + n | n - n | ...    ; constant expressions
            | x := n
            | skip ; c
            | if true then c else c
            | if false then c else c
            | while b do c
(Ctxt) E ::= []                      ; the hole containing current redex
            | E + e | E - e | ...
            | n + E | n - E | ...
            | x := E
            | E ; c
            | if E then c else c
then every program is in the form of 'E[r]' (recursively a tree), each step we evalute the current redex in the nearest context, e.g:

x := 1 + (2 + 8)
  r = (2 + 8)
  E = (x := 1 + [])
  E[r] = (x := 1 + (2 + 8))
by local reduction rule, (2 + 8, σ) -> (10, σ)
by global reduction rule, (E[2 + 8], σ) -> (E[10], σ), ie. (x := 1 + (2 + 8), σ) -> ((x := 1 + 10), σ)
```

## Big-step

big-step semantics (a.k.a. natural semantics) describes the overall result of the execution

```
(n, σ) ⇓ [n]
if σ(x) = n, then (x, σ) ⇓ [n]
if (e1, σ) ⇓ n1 and (e2, σ) ⇓ n2, then (e1 op e2, σ) ⇓ n1 [op] n2

(true, σ) ⇓ true
(false, σ) ⇓ false
if (b1, σ) ⇓ false, then (b1 and b2, σ) ⇓ false  ; short-circuit
if (b1, σ) ⇓ false and (b2, σ) ⇓ true, then (b1 and b2, σ) ⇓ false  ; non-short-circuit, other rules alike

if (e, σ) ⇓ [n], then (x := e, σ) ⇓ σ{x~>n}

(skip, σ) ⇓ σ

if (c0, σ) ⇓ σ' and (c1, σ') ⇓ σ'', then (c0 ; c1, σ) ⇓ σ''

if (b, σ) ⇓ true and (c0, σ) ⇓ σ', then (if b then c0 else c1, σ) ⇓ σ'
if (b, σ) ⇓ false and (c1, σ) ⇓ σ', then (if b then c0 else c1, σ) ⇓ σ'

if (b, σ) ⇓ false, then (while b do c, σ) ⇓ σ
if (b, σ) ⇓ true and (c, σ) ⇓ σ' and (while b do c, σ') ⇓ σ'', then (while b do c, σ) ⇓ σ''

if (e, σ) ⇓ [n] and (c, σ{x~>n}) ⇓ σ', then (newvar x := e in c, σ) ⇓ σ'{x~>σ(x))}
```

Some facts about '⇓':

  - Determinism: if (e, σ) ⇓ n and (e, σ) ⇓ n', then n = n'
  - Totality: forall e σ, exists n, st. (e, σ) ⇓ [n]  (ie. no exception or dead loop)
  - Equivalence to small-step semantics: (e, σ) ⇓ [n] *iff* (e, σ) ->* (n, σ)

## Apllication on pure LC

```
[Small-step SOS]
(Terms) M, N ::= x | λx. M | M N

(λx. M) N -> M[N/x]
if M -> M', then M N -> M' N
if N -> N', then M N -> M N'
if M -> M', then λx. M -> λx. M'

[Small-step Context]
(Terms) M, N ::= x | λx. M | M N
(Redex) r ::= (λx. M) N
(Context) E ::= [] | λx. E | E N | M E

(λx. M) N -> M[N/x]           ; local reduction rule
if r -> M, then E[r] -> E[M]  ; global reduction rule

[Big-step]
(Terms) M, N ::= x | λx. M | M N

x ⇓ x
if M ⇓ M', then λx. M ⇓ λx. M'
if M ⇓ λx. M' and N ⇓ N' and M'[N'/x] ⇓ P, then M N ⇓ P
```


# Hoare logic

Floyd-Hoare Logic is a method of reasoning mathematically about **imperative programs**

## Hoare triple/notation

So-called Assertion Language to wrap up the programming language:

```
{p}c{q}    ; partial correctness specification
           ;   (initial state satisfying p) -> (c terminates -> final state satisfies q)
[p]c[q]    ; total correctness specification
           ;   (initial state satisfying p) -> (c terminates && final state satisfies q)

*p and q are assertions, p is called precondition, and q is called postcondition
  total correctness = termination + partial correctness
*termination is not always straightforward to show:
  while x > 1 do if odd(x) then x := (3 * x) + 1 else x := x / 2

[example specs]
{x = 1} x := x + 1 {x = 2}                      ; valid
{x = 1} x := x + 1 {x = 3}                      ; invalid
{x - y > 3} x := x - y {x > 2}                  ; valid
[x - y > 3] x := x - y [x > 2]                  ; valid
{x <= 10} while x != 10 do x := x + 1 {x = 10}  ; valid
[x <= 10] while x != 10 do x := x + 1 [x = 10]  ; valid
{true} while x != 10 do x := x + 1 {x = 10}     ; valid
[true] while x != 10 do x := x + 1 [x = 10]     ; invalid
{x = 1} while true do skip {x = 2}              ; valid

[logical/ghost variables]
{x = x0 ^ y = y0} r := x ; x := y ; y := r {x = y0 ^ y = x0}
*here 'x0' and 'y0' holds constant value and not occurs in program, is called ghost variable, often used to memorize initial constant values

[some very special specs]
{true}c{q}  ; whenever c terminates, q always holds
[true]c[q]  ; c always terminates, more over it ends in a state where q holds
{p}c{true}  ; says nothing, valid for any (p, c)
[p]c[true]  ; if p holds, then c must terminate
```

Specifications rules for our simple imperative language:

```
{p[e/x]} x := e {p}                    ; AS
{p} x := e {∃v. x = e[v/x] ∧ p[v/x]}  ; AS-FW, renaming x to v, evalute e then assign it to x

if p => r and {r}c{q}, then {p}c{q}    ; SP
if {p}c{r} and r => q, then {p}c{q}    ; WC
if p => p' and {p'}c{q'} and q' => q, then {p}c{q}  ; CONSEQ

if {p}c1{r} and {r}c2{q}, then {p}c1 ; c2{q}  ; SC

{p}skip{p}  ; SK

if {p∧b}c1{q} and {p∧~b}c2{q}, then {p}if b then c1 else c2{q}  ; CD
if {p}c{q} and {p'}c{q'}, then {p∧p'}c{q∧q'}  ; CA
if {p}c{q} and {p'}c{q'}, then {p∨p'}c{q∨q'}  ; DA

if {i∧b}c{i}, then {i}while b do c{i∧~b}      ; WHP, i is called loop invariant
if [i∧b∧e=x0]c[i∧(e<x0>)] and i∧b => e>=0, then [i]while b do c[i∧~b]  ; WHT, e is called loop variant which always decreases in each iteration, where its initial value x0 ∉ fv(c) ∪ fv(e) ∪ fv(i) ∪ fv(b)

if {p}c{q} and [p]c[true], then [p]c[q]
if {p}c{q} and (c contains no while commands), then [p]c[q]
if [p]c[q], then {p}c{q}
if [p]c[q], then [p]c[true]
```

How to find loop invariants is a work of intelligence, be ware of those relations with variables or expressions which:

  - holds initailly
  - holds while b holds (=true)
  - still holds even when b become false, and result is established right on that time
  - usually is an equation about what has been done so far together with what remains to be
done


## Reasoning/Proving examples

```
[assignment]
  {x=n} x := x+1 {x=n+1}
1. x=n => x+1=n+1                     ; predicate logic
2. {x+1=n+1} x := x+1 {x=n+1}       ； AS
3. {x=n} x := x+1 {x=n+1}           ; SP 1,2

  {r=x} z := 0 {r=x+(y*z)}
1. r=x => r=x∧0=0                  ; predicate logic
2. {r=x∧0=0} z := 0 {r=x∧z=0}   ; AS
3. {r=x} z := 0 {r=x∧z=0}          ; SP 1,2
4. r=x∧z=0 => r=x+(y*z)            ; predicate logic
5. {r=x} z := 0 {r=x+(y*z)}           ; WC 3,4

  {y>3} x := 2*y; x := x-y {x>=4}
1. {x-y>=4} x := x-y {x>=4}           ; AS
2. {2*y-y>=4} x := 2*y {x-y>=4}       ; AS
3. y>3 => 2*y-y>=4                    ; predicate logic
4. {y>3} x := 2*y {x-y>=4}            ; SP 1,2
5. {y>3} x := 2*y; x := x-y {x>=4}    ; SC 1,4

[loop]
  {x<=10} while x!=10 do x := x+1 {x=10}    ; notice that i is x<=10, b is x!=10
1. {x+1<=10} x := x+1 {x<=10}               ; AS
2. x<=10∧x!=10 => x+1<=10                  ; predicate logic
3. {x<=10∧x!=10} x := x+1 {x<=10}          ; SP 1,2
4. {x<=10} while x!=10 do x := x+1 {x<=10∧~(x!=10)}   ; WHP 3
5. x<=10∧~(x!=10) => x=10                  ; predicate logic
6. {x<=10} while x!=10 do x := x+1 {x=10}   ; WC 4,5

  {true} while x!=10 do skip {x=10}
1. {true∧x!=10} skip {true∧x!=10}         ; SK
2. true∧x!=10 => true                      ; predicate logic
3. {true∧x!=10} skip {true}                ; WC 1,2
4. {true} while x!=10 do skip {true∧~(x!=10)}  ; WHP 3
5. true∧~(x!=10) => x=10                   ; predicate logic
6. {true} while x!=10 do skip {x=10}        ; WC 4,5

  [x<=10] while x!=10 do x := x+1 [x=10]    ; notice that i is x<=10, b is x!=10, e is 10-x
1. {x+1<=10∧10-(x+1)<z} x := x+1 {x<=10∧10-x<z}     ; AS
2. x<=10∧x!=10∧10-x=z => x+1<=10∧10-(x+1)<z
3. {x<=10∧x!=10∧10-x=z} x := x+1 {x<=10∧10-x<z}    ; SP 1,2
4. x<=10∧x!=10 => 10-x>=0
5. [x<=10] while x!=10 do x := x+1 [x<=10∧~(x!=10)]  ; WHT 3,4
6. x<=10∧~(x!=10) => x=10
7. [x<=10] while x!=10 do x := x+1 [x=10]   ; WC 5,6
```

## Annotation & Automated program verification

Where to insert annotation:

  - before statements except assignments
  - before while body

## Soundness & Completeness

```
|- {p}c{q} means that there exists a derivation sequence following the rules (ie. provable)
|= {p}c{q} means the semantical meaning of {p}c{q} (ie. meaningful)

Soundness of the program logic: every provable is meaningly true
  - if |- {p}c{q}, then |= {p}c{q}
  - if |- [p]c[q], then |= [p]c[q]
Completeness of the program logic: every meaningly true is provable
  - if |= {p}c{q}, then |- {p}c{q}
  - if |= [p]c[q], then |- [p]c[q]
  - *sometimes might violate Godel's incompleteness theorem

Hoare logic is both sound and complete, provided that the underlying logic is!
*BUT often, the underlying logic is sound but incomplete...
 consider '|= {true}c{false} iff c does not halt', but HALT is undecidable
 so the regular hoare logic based on **predicate logic** is incomplete (caused by rule SP and WC), but relative-complete

[Cook 1978] Hoare logic is relative-complete:
  if |= {p}c{q}, then {p | (|=p)} |- {p}c{q}
```


# Separation logic

Separation logic is a hoare logic extension adding support for pointers

Exented the programming language syntax and corresponding semantics to support pointers:

```
(Comm) c ::= ...
           | x := cons(e1, e2, ..., en)   ; allocation, address is random
           | dispose(e)                   ; deallocation
           | x := [e]                     ; lookup
           | [e] := e                     ; mutation
(Var)   v ∈ Nat
(Loc)   l ∈ Nat
(Store) s ∈ Var -> Nat
(Heap)  h ∈ Loc ->fin Nat
(State) σ ∈ (s, h)

if [[e]](intexpr)s ∉ dom(h), then (x := [e], (s, h)) -> abort
if h([[e]](intexpr)s) = n, then (x := [e], (s, h)) -> (skip, (s{x~>n}, h))

if [[e]](intexpr)s ∉ dom(h), then ([e] := e', (s, h)) -> abort
if [[e]](intexpr)s = l and l ∈ dom(h) and [[e']](intexpr)s = n
    then ([e] := e', (s, h)) -> (skip, (s, h{l~>n}))

if [[e1]](intexpr)s = n1 and [[e2]](intexpr)s = n2 and {l,l+1} ∩ dom(h) = Ø
    then (x := cons(e1, e2), (s, h)) -> (skip, (s{x~>l}, h{l~>e1, l+1~>e2}))
```

## Assertion Syntax and Abbreviations

```
(Assert) p ::= emp | e1 |-> e2 | p1 * p2 | p1 -* p2
             | b | ~p | p1 ∧ p2 | p1 ∨ p2 | p1 => p2
             | ∀x. p | ∃x. p

emp         ; empty heap
e |-> e'    ; singleton heap containing one cell at address e with content e'
p1 * p2     ; separation conjuction, the heap can be split into two disjoint heap p1 and p2
p1 -* p2    ; separation implication, p1 is a subset of p2
e |-> -     ; ∃x. e |-> x, means that heap has exactly one element but we ignore its value
e \-> e'    ; e |-> e' * true, means that heap at least has one element e'
e |-> e1, e2, ..., en  ; means heap continously stores values (e1, e2, ..., en), head address is e
e \-> e1, e2, ..., en  ; means heap at least continously stores these values
h0 |_ h1    ; heap h0 and h1 have disjoint domains, '|_' is the symbol of 'perpendicular to' 
h0 · h1     ; the union of heap h0 and h1 with disjoint domains

[exmaples]
(1) x = { (3,y) }, y = { (3,x) }     ; (val, next)
(2) x = y = { (3,x/y) }
x |-> 3,y * y |-> 3,x    ; (1)
x |-> 3,y ∧ y |-> 3,x   ; (2)
x \-> 3,y ∧ y \-> 3,x   ; (1) or (2), and may contain other cells
```

## Semantics of Assertions

```
s, h |= emp iff dom h = {}
s, h |= e |-> e' iff dom h = {[[e]](exp)s} and h([[e]](exp)s) = [[e']](exp)s
s, h |= p0 * p1 iff ∃h0 h1, h0 |_ h1 and h0 · h1 = h and s, h0 |= p0 and s, h1 |= p1
s, h |= p0 -* p1 iff ∀h', (h' |_ h and s, h' |= p0) implies s, h · h' |= p1

s, h |= b iff [[b]](boolexpr)s = true
s, h |= ~p iff s, h |= p is false
s, h |= p0∧p1 iff s, h |= p0 and s, h |= p1   ; ∨, =>, <=> is alike
s, h |= ∀x. p iff ∀x∈Z. [s | v:x], h |= p
s, h |= ∃x. p iff ∃x∈Z. [s | v:x], h |= p

valid: if s, h |= p holds for all (s, h)
satisfiable: if s, h |= p holds for some (s, h)
```

Assertion classed by properties:

  - Pure assertions: forall s h h', s, h |= p iff s, h' |= p (quick check: not contain emp, |->, \->)
  - Strict exact assertions: forall s h h', s, h |= p and s, h' |= p implies h = h'
  - Precise assertions: forall s h, there is at most one h' ⊆ h such that s, h' |= p
  - Intuitionstic assertions: forall s h h', (h ⊆ h' and s, h |= p) implies s, h' |= p

## Specifications and Inference Rules

```
[examples-valid]
           {emp} x := cons(1,2) {x|->1,2}
       {x|->1,2} y := [x]       {x|->1,2 ∧ y=1}
{x|->1,2 ∧ y=1} [x + 1] := 3   {x|->1,3 ∧ y=1}
{x|->1,3 ∧ y=1} dispose x      {x+1|->3 ∧ y=1}

[inference rules]
SP and WC still holds, but Rule of Constancy fails
    "if {p}c{q}, then {p∧r}c{q∧r}, where c doesn't modify fv(r)"
let p = x |-> -
    c = [x] := 4
    q = x |-> 4
    r = y |-> 3
the rule fails when x = y
*thus instead we have frame rule

[Frame Rule (O'Hearn)]
  FR: "if {p}c{q}, then {p*r}c{q*r}, where c doesn't modify fv(r)"
*this rule is the key to "local reasoning" about heap

[local reasoning]
  - the set of variables and heap cells that used by a command is called its *footprint*
  - if {p}c{q} is valid, then p asserts that the heap contains all cells in footprint of c (excluding newly allocated by c)
  - if p asserts the heap contains ONLY cells in the footprint of c, then {p}c{q} is a *local specification*
  - if c' contains c, it may have a larger footprint described by p*r, then thr frame rule is needed to move from {p}c{q} to {p*r}c{q*r}

[example-FR]
            {list a i} rev_list() {list a_rev j}
------------------------------------------------------------
 {list a i * list b k} rev_list() {list a_rev j * list b k}
*suppose rev_list() doesn't modify b or k

[soundness of FR]
these are equivalent expressions:
  - FR is *sound* for both partial and total correctness
  - the programming language satisfies *safety monotonicity* and *frame property*
  - the programming language satisfies *locality*

......
(other inference rules omitted: MUL, DISL, CONSL, LKL)
```

## List and List Segments

Notation for a sequence:

  - ε: empty seq
  - [a]: a single-element seq containg a
  - α*β: composition of α followed by β
  - α†: reverse
  - #α: cardinality/length
  - αi: the i-th element

Single-linked list:

```
[notation syntax]
list α i: a list containing a sequence α whose head pointer is i

[recursive definition]
    list ε i ::= emp ∧ i = nil
list (a*α) i ::= ∃j. [i |-> (a,j)] * [list α j]
```

Single-linked list segments:

```
[notation syntax]
lseg α (i,j): a list containing a sequence α whose head pointer is i and tail pointer is j

[recursive definition]
    lseg ε (i,j) ::= emp ∧ i = j
lseg (a*α) (i,j) ::= ∃k. [i |-> (a,k)] * [lseg α (k,j)]

[properties]
  lseg a (i,j) == i |-> (a,j)
lseg α*β (i,k) == ∃j. lseg α (i,j) * lseg β (j,k)
lseg α (i,nil) == list α i
```
