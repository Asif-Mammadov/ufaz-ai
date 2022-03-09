% s(CASP) Programming
:- use_module(library(scasp)).
% Uncomment to suppress warnings
%:- style_check(-discontiguous).
%:- style_check(-singleton).
%:- set_prolog_flag(scasp_unknown, fail).

% Your program goes here


/** <examples> Your example queries go here, e.g.
?- ? p(X).
*/

parent(X, Y) :- child(Y, X).
father(X, Y) :- parent(X, Y), male(X).
mother(X, Y) :- parent(X, Y), female(X).
son(X, Y) :- child(X, Y), male(X).
daughter(X, Y) :- child(X, Y), female(X).

grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
grandfather(X, Y) :- grandparent(X, Y), male(X).
grandmother(X, Y) :- grandparent(X, Y), female(X).
grandson(X, Y) :- grandparent(Y, X), male(X).
granddaughter(X, Y) :- grandparent(Y, X), female(X).

sibling(X, Y) :- father(F, X), father(F, Y), X\=Y.
sibling(X, Y) :- mother(M, X), mother(M, Y), X\=Y.
brother(X, Y) :- sibling(X, Y), male(X).
sister(X, Y) :- sibling(X, Y), female(X).

uncle(X, Y) :- brother(X, Z), parent(Z, Y).
aunt(X, Y) :- sister(X, Z), parent(Z, Y).

male_cousin(X, Y) :- sibling(S1, S2), son(X, S1), child(Y, S2).
female_cousin(X, Y) :- sibling(S1, S2), daughter(X, S1), child(Y, S2).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

male(brandon).
male(eddard).
male(rickard).
male(benjen).
male(jon).
male(robb).
male(hoster).
male(robin).

female(catelyn).
female(arya).
female(lyanna).
female(lysa).

% zoomer starks
child(brandon, eddard).
child(brandon, catelyn).
child(arya, eddard).
child(arya, catelyn).
child(robb, eddard).
child(robb, catelyn).

% boomer starks 
child(eddard, rickard).
child(benjen, rickard).
child(lyanna, rickard).

% wife side
child(catelyn, hoster).
child(lysa, hoster).
child(robin, lysa).

% others
child(jon, lyanna).