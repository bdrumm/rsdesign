"""Failure-scenario training harness: every failure becomes a parameterised family of generated
cases that joins the regression gate and drives tuning (docs/SCENARIOS.md).

* :mod:`dt.scenarios.spec`     -- ScenarioFamily / ScenarioCase / Criterion
* :mod:`dt.scenarios.sources`  -- IR-generated, HTML(DOM)-generated and real-crop cases
* :mod:`dt.scenarios.measures` -- pipeline stage subsets + structural / fidelity measures in a ROI
* :mod:`dt.scenarios.registry` -- auto-discovered families (``dt/scenarios/families``, ``$DT_HOME``)
* :mod:`dt.scenarios.runner`   -- run a family on seeds: pass rate, margins, worst cases + images
* :mod:`dt.scenarios.suite`    -- the active suite (``knowledge/scenarios.json``), promote, gate
* :mod:`dt.scenarios.mine`     -- translate run dirs -> real-crop cases
* :mod:`dt.scenarios.train`    -- the self-tuning loop (``dt train``) + leave-one-family-out report
* :mod:`dt.scenarios.cli`      -- ``dt scenario|train|learn`` and ``dt bench --scenarios``
"""
