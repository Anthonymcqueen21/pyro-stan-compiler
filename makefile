CMD=g++  -D FUSION_MAX_VECTOR_SIZE=12 -DBOOST_RESULT_OF_USE_TR1 -DBOOST_NO_DECLTYPE -DBOOST_DISABLE_ASSERTS -Os -ftemplate-depth-256 -Wno-unused-function -Wno-uninitialized -I stan/src -I stan/lib/stan_math/ -I stan/lib/stan_math/lib/eigen_3.3.3 -I stan/lib/stan_math/lib/boost_1.65.1 -I stan/lib/stan_math/lib/cvodes_2.9.0/include


all: stan2pyro/bin/stan2pyro

clean:
	-rm -rf stan2pyro/bin/ stan2pyro/build/

stan2pyro/bin/stan2pyro: stan2pyro/build/ast_def.o stan2pyro/build/bare_type_grammar_inst.o stan2pyro/build/expression07_grammar_inst.o stan2pyro/build/expression_grammar_inst.o stan2pyro/build/functions_grammar_inst.o stan2pyro/build/indexes_grammar_inst.o stan2pyro/build/program_grammar_inst.o stan2pyro/build/semantic_actions_def.o stan2pyro/build/statement_2_grammar_inst.o stan2pyro/build/statement_grammar_inst.o stan2pyro/build/term_grammar_inst.o stan2pyro/build/var_decls_grammar_inst.o stan2pyro/build/whitespace_grammar_inst.o stan2pyro/build/stan2pyro.o build_dir
	$(CMD) -o stan2pyro/bin/stan2pyro stan2pyro/build/*.o

exe: stan2pyro/build/stan2pyro.o
	$(CMD) -o stan2pyro/bin/stan2pyro stan2pyro/build/*.o

stan2pyro/build/ast_def.o: stan/src/stan/lang/ast_def.cpp build_dir
	$(CMD) -c -o stan2pyro/build/ast_def.o stan/src/stan/lang/ast_def.cpp

stan2pyro/build/bare_type_grammar_inst.o: stan/src/stan/lang/grammars/bare_type_grammar_inst.cpp build_dir
	$(CMD) -c -o stan2pyro/build/bare_type_grammar_inst.o stan/src/stan/lang/grammars/bare_type_grammar_inst.cpp

stan2pyro/build/expression07_grammar_inst.o: stan/src/stan/lang/grammars/expression07_grammar_inst.cpp build_dir
	$(CMD) -c -o stan2pyro/build/expression07_grammar_inst.o stan/src/stan/lang/grammars/expression07_grammar_inst.cpp

stan2pyro/build/expression_grammar_inst.o: stan/src/stan/lang/grammars/expression_grammar_inst.cpp build_dir
	$(CMD) -c -o stan2pyro/build/expression_grammar_inst.o stan/src/stan/lang/grammars/expression_grammar_inst.cpp

stan2pyro/build/functions_grammar_inst.o: stan/src/stan/lang/grammars/functions_grammar_inst.cpp build_dir
	$(CMD) -c -o stan2pyro/build/functions_grammar_inst.o stan/src/stan/lang/grammars/functions_grammar_inst.cpp

stan2pyro/build/indexes_grammar_inst.o: stan/src/stan/lang/grammars/indexes_grammar_inst.cpp build_dir
	$(CMD) -c -o stan2pyro/build/indexes_grammar_inst.o stan/src/stan/lang/grammars/indexes_grammar_inst.cpp

stan2pyro/build/program_grammar_inst.o: stan/src/stan/lang/grammars/program_grammar_inst.cpp build_dir
	$(CMD) -c -o stan2pyro/build/program_grammar_inst.o stan/src/stan/lang/grammars/program_grammar_inst.cpp

stan2pyro/build/semantic_actions_def.o: stan/src/stan/lang/grammars/semantic_actions_def.cpp build_dir
	$(CMD) -c -o stan2pyro/build/semantic_actions_def.o stan/src/stan/lang/grammars/semantic_actions_def.cpp

stan2pyro/build/statement_2_grammar_inst.o: stan/src/stan/lang/grammars/statement_2_grammar_inst.cpp build_dir
	$(CMD) -c -o stan2pyro/build/statement_2_grammar_inst.o stan/src/stan/lang/grammars/statement_2_grammar_inst.cpp

stan2pyro/build/statement_grammar_inst.o: stan/src/stan/lang/grammars/statement_grammar_inst.cpp build_dir
	$(CMD) -c -o stan2pyro/build/statement_grammar_inst.o stan/src/stan/lang/grammars/statement_grammar_inst.cpp

stan2pyro/build/term_grammar_inst.o: stan/src/stan/lang/grammars/term_grammar_inst.cpp build_dir
	$(CMD) -c -o stan2pyro/build/term_grammar_inst.o stan/src/stan/lang/grammars/term_grammar_inst.cpp

stan2pyro/build/var_decls_grammar_inst.o: stan/src/stan/lang/grammars/var_decls_grammar_inst.cpp build_dir
	$(CMD) -c -o stan2pyro/build/var_decls_grammar_inst.o stan/src/stan/lang/grammars/var_decls_grammar_inst.cpp

stan2pyro/build/whitespace_grammar_inst.o: stan/src/stan/lang/grammars/whitespace_grammar_inst.cpp build_dir
	$(CMD) -c -o stan2pyro/build/whitespace_grammar_inst.o stan/src/stan/lang/grammars/whitespace_grammar_inst.cpp

stan2pyro/build/stan2pyro.o: stan2pyro/stan2pyro.cpp build_dir
	$(CMD) -c -o stan2pyro/build/stan2pyro.o stan2pyro/stan2pyro.cpp

build_dir:
	mkdir -p stan2pyro/build/
	mkdir -p stan2pyro/bin/
