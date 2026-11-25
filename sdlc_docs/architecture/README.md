# ACID architecture diagrams

[`workspace.dsl`](workspace.dsl) has two views: the C4 container view of the three execution paths and a Snakemake rule view of file handoffs. The [architecture analysis](docs/overview.md) maps source components and stages, and records the current refactor constraints.

## View locally with Docker

From the repository root:

```bash
docker pull structurizr/structurizr:2026.09.19
docker run --rm -it -p 127.0.0.1:8080:8080 \
  -v "$PWD/sdlc_docs/architecture:/usr/local/structurizr" \
  structurizr/structurizr:2026.09.19 local
```

Open <http://localhost:8080>, then select **Diagrams**. Use the diagram's information button to see the notation key. Stop the container with Ctrl+C. After editing `workspace.dsl`, refresh the browser. Structurizr may create `workspace.json`, `.structurizr/`, and `down/` in this directory; those local files are ignored by Git. The DSL views use automatic layout.

Validate the workspace without starting the web interface:

```bash
docker run --rm \
  -v "$PWD/sdlc_docs/architecture:/usr/local/structurizr" \
  structurizr/structurizr:2026.09.19 \
  validate -workspace /usr/local/structurizr/workspace.dsl
```

