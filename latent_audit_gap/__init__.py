"""Dissociated-model construction and intervention-based safety evaluation.

Three stages, run in order per architecture: ``harmful`` trains the harmful pole,
``dissociated`` builds and audits the dissociated model, and ``intervention`` measures how far
bounded interventions reach into it. ``config`` holds every setting.
"""
