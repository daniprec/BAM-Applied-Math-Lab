# Slide Template Instructions

Use this guide when you create or revise a Quarto slide deck for Applied Math Lab.

## Source of truth

- Start from the matching module pages in `modules/`.
- Use the legacy `pptx` only to recover the original pacing, emphasis, or classroom prompts.
- If the module and the old slides disagree, keep the module content.

## Deck shape

- Aim for 8 to 12 content slides plus the title slide.
- Keep one main idea per slide.
- Use short bullets, equations, code snippets, or comparison cards instead of long paragraphs.
- End with a slide that points students to the full module pages.

## Recommended sequence

1. Session framing: why the topic matters in the course arc.
2. Session map: 3 to 5 concepts or case studies.
3. Core mathematics: the main equations, rules, or update scheme.
4. Computational workflow: the Python tools and implementation pattern.
5. Case-study slides: the most instructive models or datasets.
6. Exploration slide: what students should vary or test.
7. Module links: where to read the full material.

## Writing rules

- Prefer direct sentences and short labels.
- Keep bullets concrete. Name the parameter, object, or algorithm.
- Use equations when they clarify the model faster than prose.
- Keep code blocks short, usually no more than 12 lines.
- If a figure already exists in `img/`, reuse it.
- If a concept needs more detail, link to the relevant module page instead of overloading the slide.

## Design rules

- Use the shared reveal.js settings in `slides/_metadata.yml`.
- Use the shared styling in `slides/amlab-slides.css`.
- Use `##` headings for real slide breaks. Inner panel labels are normalized by the shared slide filter so Reveal only sees actual slides.
- Favor `card`, `grid-2`, `grid-3`, `module-links`, `inverse`, and `section-break` layouts.
- Use `section-break` only for high-level transitions.
- Use `inverse` for emphasis slides with one core message.

## Quarto frontmatter

```yaml
---
title: "Session Title"
subtitle: "Short descriptive subtitle"
---
```

The shared metadata already supplies the reveal.js format, slide numbering, transitions, and disabled execution for code examples.

## Minimal slide scaffold

```markdown
## Session Arc

::: {.grid-3}
::: {.card}
### Concept
One sentence on what students should understand.
:::

::: {.card}
### Method
One sentence on the computational workflow.
:::

::: {.card}
### Outcome
One sentence on what the model reveals.
:::
:::

## Core Model

$$
\dot x = f(x, y; \theta)
$$

- State what each variable represents.
- State which parameter matters most.

## Full Module Pages

::: {.module-links}
[Overview](../modules/topic/index.qmd)
[Main case study](../modules/topic/case-study.qmd)
[Assignment](../modules/topic/assignment.qmd)
:::
```
