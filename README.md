## README

- [English](README.md)
- [Russian](README-ru.md)

## Project Tree

```
.
|-- docker-compose.yml  # Docker-compose configuration
|-- LICENSE             # Repository license
|-- README*.md          # Repository information file
`-- rep                 # Folder report
    |-- a               # Folder ready to "Appendix A"
    |   |-- a.pdf       # Compiled "Appendix A" in *.pdf
    |   `-- a.tex       # Source code "Appendix A" in *.tex
    |-- b               # The folder with the output of "Appendix B"
    |   |-- b.pdf       # Compiled "Appendix B" in *.pdf
    |   `-- b.tex       # Source code of "Appendix B" in *.tex
    |-- _INCLUDES       # Folder with *.tex source codes that are connected
    |   `-- *           # Any nesting of folders
    |       `-- *.tex   # Source code *.tex that connect
    |-- main            # Folder ready "Explanatory note"
    |   |-- main.pdf    # Compiled "Explanatory note" in *.pdf
    |   `-- main.tex    # The source code of the "Explanatory notes" in *.tex
    `-- _STYLES         # Folder styles *.sty
        `-- *.sty
```

## Compile PDF

Docker must be installed.

Launching the Docker VM via `docker-compose.yml`:

```bash
sudo docker-compose run latex /bin/bash
```

Once the VM has started, go to the directory with the `main.tex` (or `a.tex`, or `b.tex`) file:

```bash
cd /content/rep/main
# cd /content/rep/a
# cd /content/rep/b
```

Compile the file `main.tex` (or `a.tex`, or `b.tex`):

```bash
pdflatex main.tex
# pdflatex a.tex
# pdflatex b.tex
```

The second time we compile the file `main.tex` (or `a.tex`, or `b.tex`), so that the links in the document work correctly:

```bash
pdflatex main.tex
# pdflatex a.tex
# pdflatex b.tex
```

The compiled `main.pdf` file in the `rep/main` folder along with the `main.tex` file.

The compiled `a.pdf` file in the `rep/a` folder along with the `a.tex` file.

The compiled `b.pdf` file in the `rep/b` folder along with the `b.tex` file.
