# How Do You Know Whether to Trust a Model's Prediction?

This is a plain-language introduction to applicability domain checking, aimed at
researchers who use models rather than people who build software for a living. If
you are comfortable with Python packaging, type hints, and dataclasses, you want
the main [README](../README.md) instead. If you just want to understand the idea,
and see it work once in front of you, this is the right page.

## The problem, stated plainly

Suppose a colleague has built a model from some data they collected, and it
predicts a number for you whenever you give it an input. The model does not know
the difference between an input that looks like the data it learned from and an
input that looks nothing like it at all. It will produce a confident-looking
number either way.

That matters because a model's accuracy was measured on data similar to what it
was trained on. Ask it about something genuinely new, something in a part of the
input space it never saw an example of, and you are no longer asking it to
interpolate between things it learned. You are asking it to guess at something
it has no real basis for, and it will still hand you back a number, with no
built-in way of telling you that this prediction is different in kind from the
ones it was actually good at.

This is a known issue, and it has a name: the model's **applicability domain** is
the region of input space its training data actually covers. A prediction made
for an input inside that region is on reasonably solid ground. A prediction made
for an input outside it is an extrapolation, and should be treated with more
caution, independent of how confident the model's output looks.

## Two ways of checking, in plain terms

There is more than one way to ask "is this input like the ones the model learned
from", and they catch different kinds of problem. This package implements two of
the most established ones.

**How far from the crowd.** Take all your training inputs and look at where they
sit as a cloud of points. Some new input might sit right in the middle of that
cloud, surrounded on all sides by training examples. Another might sit way out
past the edge, in a direction the training data barely explored at all. The
technical name for this is **leverage**, and it comes from a much older idea in
statistics: a measurement of how unusual a point is relative to the overall
shape and spread of a dataset. A high-leverage point is one that stands apart
from the general pattern of the training data, even if no single training point
happens to be nearby.

**How close is the nearest thing I have actually seen.** This is a more direct,
local question: forget the overall shape of the cloud, just ask how far away the
single nearest training example actually is. A new input can have perfectly
ordinary leverage, sitting well within the overall spread of the data, and still
happen to fall in a gap where nothing was ever actually measured nearby. This
catches a different failure than leverage does.

Used together, these two checks cover each other's blind spots. A point that
passes both is genuinely well supported by the training data in two independent
senses. A point that fails either one is worth a second look before you trust
what the model told you about it.

## Seeing it work

Here is the whole idea in four lines of code, training on 200 points scattered
around the origin in five dimensions, then checking three new points: one right
at the centre of that cloud, one an ordinary-looking point nearby, and one far
outside anything the training data looked like.

```python
import numpy as np
from adcheck import ApplicabilityDomain

rng = np.random.default_rng(0)
X_train = rng.normal(size=(200, 5))

ad = ApplicabilityDomain(knn_k=3).fit(X_train)

X_query = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],      # dead centre of the training data
    [1.0, -1.0, 0.5, 0.0, -0.5],    # an unremarkable point
    [8.0, 8.0, 8.0, 8.0, 8.0],      # far outside anything seen in training
])

result = ad.evaluate(X_query)
```

This is the actual, unedited output of running that code:

```
point               leverage   leverage_ok    knn_dist    knn_ok   in_domain
----------------------------------------------------------------------------
origin                0.0000          True      0.7339      True        True
typical point         0.0152          True      0.9767      True        True
far outlier           1.6354         False     15.4123     False       False
```

Read this a row at a time.

The **origin** has essentially zero leverage, it is sitting right at the centre
of the training data's spread, and its nearest training point is a perfectly
ordinary distance away. Both checks pass. Nothing surprising: a prediction here
is well supported.

The **typical point** is not at the exact centre, so its leverage is slightly
higher than zero, but still comfortably low, and its nearest training neighbour
is close by. Both checks pass again. This is what a normal, unremarkable query
should look like: neither number stands out.

The **far outlier** fails both checks at once, and by a wide margin. Its
leverage is more than eighteen times the threshold that separates typical points
from unusual ones, and well over a hundred times higher than the typical point's
own leverage. The nearest training point to it is roughly fifteen to twenty
times farther away than the nearest training point was for either of the other
two. Whatever a model trained on this data predicted here, that prediction is an
extrapolation, and the size of these two numbers is telling you exactly that, in
a way the model's own output never would on its own.

## What to actually do with a verdict like this

A failed check is not proof that a prediction is wrong. It is a flag that the
model had little or nothing comparable to learn from when making that particular
prediction, so its usual accuracy does not necessarily apply there. Treat it the
way you would treat a measurement taken right at the edge of an instrument's
calibrated range: not automatically wrong, but not something to build a strong
conclusion on without independent support.

A passed check is not a guarantee either. It tells you the input resembles the
training data in these two specific geometric senses. It says nothing about
whether the training data itself was good, whether the model learned the right
relationship from it, or whether some other kind of error is present. This is a
sanity check on one particular failure mode, extrapolation, not a substitute for
validating the model properly in the first place.

## Trying it on your own data

```python
from adcheck import ApplicabilityDomain

ad = ApplicabilityDomain().fit(X_train)       # your own training inputs
result = ad.evaluate(X_new)                   # whatever you want to check

print(result.in_domain)      # True or False, one per row of X_new
print(result.leverage)       # the actual numbers behind that verdict
print(result.knn_distance)
```

`X_train` and `X_new` just need to be plain numeric arrays, the same features in
the same order in both. The package makes no assumption about what those
features represent, so this works whether your model was trained on spectra,
sensor readings, survey responses, or anything else expressed as numbers.

## Further reading

The main [README](../README.md) covers installation, the full API, the exact
formulas behind both checks, and references to the published guidance this
method follows, originally developed for regulatory use in chemical risk
assessment and adopted more widely since.
