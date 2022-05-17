# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = ../docs

# Put it first so that "make" without argument is like "make help".
defhtml:
	# remove the previous files
	@$(SPHINXBUILD) -E -b dirhtml "$(SOURCEDIR)" "$(BUILDDIR)" 
	touch "$(BUILDDIR)/.nojekyll"
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -r $(BUILDDIR)/.doctrees
	rm -r $(BUILDDIR)/*
	sleep 0.2

fresh: clean defhtml

.PHONY: fresh defhtml help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" -b html "$(BUILDDIR)" $(SPHINXOPTS) $(O)
