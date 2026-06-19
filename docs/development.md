# Development

Sake development combines Python source validation, documentation generation, MkDocs build checks, and source-driven API documentation. The development workflow is designed to keep analytical behavior, documentation pages, and API reference output aligned.

## 🧭 Purpose

Define the setup, validation, build, troubleshooting, and publishing workflow for maintaining Sake documentation and related source files.

## 🐍 Environment Setup

Run the development workflow from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 📦 Documentation Dependencies

| Package                | Role                                                      |
|------------------------|-----------------------------------------------------------|
| `mkdocs`               | Static documentation site generator.                      |
| `mkdocs-material`      | Material theme, navigation, search, and dark-mode layout. |
| `mkdocstrings[python]` | Source-driven API reference generation.                   |
| `mkdocs-autorefs`      | Cross-reference support for generated API documentation.  |
| `pymdown-extensions`   | Enhanced Markdown features used by Material for MkDocs.   |

## 🧪 Validation Workflow

Run these checks before committing documentation changes.

```powershell
python -m compileall .
mkdocs build
```

Use local preview for visual review.

```powershell
mkdocs serve
```

## 🧾 Documentation File Structure

```text
docs/
  index.md
  architecture.md
  data-sources.md
  model-evaluation.md
  development.md
  user-guide/
  api/
  assets/
    css/
      sake.css
    js/
      sake.js
```

## 🧱 Development Responsibilities

| Area           | Standard                                                                                        |
|----------------|-------------------------------------------------------------------------------------------------|
| Source files   | Preserve executable behavior unless a behavior change is explicitly required.                   |
| Docstrings     | Use Google-style sections compatible with mkdocstrings and griffe.                              |
| Markdown pages | Keep pages task-oriented, source-aligned, and build-safe.                                       |
| API pages      | Use mkdocstrings directives that match importable Python modules.                               |
| CSS            | Preserve dark-mode readability and dark blue header styling.                                    |
| JavaScript     | Provide progressive usability enhancements without analytics, popups, or visible path metadata. |
| Navigation     | Include every generated documentation page in `mkdocs.yml`.                                     |

## ✅ Google-Style Docstring Rules

| Section       | Rule                                                                                        |
|---------------|---------------------------------------------------------------------------------------------|
| `Purpose:`    | Describe the object’s role, workflow stage, state used or modified, and analytical purpose. |
| `Args:`       | Document only real parameters from the signature.                                           |
| `Attributes:` | Document public class attributes without parenthesized type clutter.                        |
| `Returns:`    | Include a type and description for meaningful return values.                                |
| `Raises:`     | Document explicit or re-raised exceptions with complete sentences.                          |
| `Notes:`      | Capture implementation constraints, assumptions, or analytical cautions.                    |
| `Examples:`   | Provide compact examples only when they clarify usage.                                      |

## 🧯 Griffe Warning Controls

| Warning Type                       | Cause                                                 | Correction                                                     |
|------------------------------------|-------------------------------------------------------|----------------------------------------------------------------|
| Failed `name: description` parsing | Malformed `Args:` or `Attributes:` entry.             | Use `name: Description.` for each entry.                       |
| Missing return type                | Function lacks annotation and `Returns:` type.        | Add a return annotation or explicit return type in `Returns:`. |
| Import failure                     | API directive references a module that cannot import. | Fix module import or update the directive.                     |
| Missing nav file                   | `mkdocs.yml` references a missing page.               | Create the page or remove the nav entry.                       |
| Unlisted page                      | Markdown file exists outside nav.                     | Add the page to nav or remove the file.                        |

## 🔄 Recommended Change Sequence

1. Update source code or Markdown files.
2. Run Python compile checks.
3. Run `mkdocs build`.
4. Fix import, nav, Markdown, or griffe warnings.
5. Run `mkdocs serve`.
6. Review navigation, tables, code blocks, API pages, and search.
7. Commit only build-safe files.

## 🧪 Source Validation Commands

| Command                                   | Purpose                                             |
|-------------------------------------------|-----------------------------------------------------|
| `python -m compileall .`                  | Confirms Python files compile.                      |
| `python -c "import app; print('app ok')"` | Confirms a module imports.                          |
| `mkdocs build`                            | Builds the documentation site and exposes warnings. |
| `mkdocs serve`                            | Runs the local documentation preview server.        |

## 🧩 API Page Validation

Each API Markdown page should reference an importable module.

```markdown
# Module Name

::: module_name
```

Before adding or keeping a directive, confirm the module imports.

```powershell
python -c "import module_name; print('module_name ok')"
```

## 🎨 Asset Validation

| Asset                      | Validation                                                                      |
|----------------------------|---------------------------------------------------------------------------------|
| `docs/assets/css/sake.css` | Confirm dark theme, dark blue header, table readability, and responsive layout. |
| `docs/assets/js/sake.js`   | Confirm no path metadata renders on pages.                                      |
| `docs/images/`             | Confirm referenced images exist before linking them.                            |

## 🚀 GitHub Pages Build Notes

| Item             | Requirement                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `site_url`       | Match the GitHub Pages URL.                                                 |
| `repo_url`       | Match the repository URL.                                                   |
| `mkdocs.yml`     | Include complete navigation and asset paths.                                |
| Published branch | Use the repository’s configured GitHub Pages branch or deployment workflow. |
| Build result     | Publish only after `mkdocs build` completes without avoidable warnings.     |

## ✅ Release Checklist

- Virtual environment activates successfully.
- Dependencies install from `requirements.txt`.
- Python source compiles.
- MkDocs build completes.
- API pages import successfully.
- No missing navigation files remain.
- No unlisted Markdown pages remain.
- CSS and JavaScript paths match `mkdocs.yml`.
- Local preview renders navigation, tables, code blocks, and API pages correctly.
