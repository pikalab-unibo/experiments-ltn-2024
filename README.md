# LTN_Imp


## Project structure
Overview:

```bash
<root directory>
├── ltn_imp/             # main package (should be named after your project)
│   ├── __init__.py         # python package marker
│   └── __main__.py         # application entry point
│   └── fuzzy_operators     # folder for all of the fuzzy operators
│   └── parsing             # folder for the parser utilizing NLTK Logic and all the needed files
│   
├── test/                   # test package contains unit tests
│   ├── parsing_tests       # folder for unit tests for parser
│   └── learning_tests      # folder for unit tests for fuzzy operators in optimization
│   
├── .github/                # configuration of GitHub CI
│   └── workflows/          # configuration of GitHub Workflows
│       └──  check.yml       # runs tests on multiple OS and versions of Python
│
├── LICENSE                 # license file (Apache 2.0 by default)
├── pyproject.toml          # declares build dependencies
└── poetry.toml             # Poetry settings
```