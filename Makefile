.PHONY: setup check test build preview proof archive-verify
setup check test build preview proof archive-verify:
	node scripts/toolchain.mjs $@
