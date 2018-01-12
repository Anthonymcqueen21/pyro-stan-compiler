
# python makefile_gen.py > makefile
# git submodule update --init --recursive

DEBUG = False
DBG_OPTS = ""
if DEBUG: #TODO: command-line should give this
    DBG_OPTS = "-g -Wall -Werror"

sources = [
    "stan/src/stan/lang/ast_def.cpp",
    "stan/src/stan/lang/grammars/bare_type_grammar_inst.cpp",
    "stan/src/stan/lang/grammars/expression07_grammar_inst.cpp",
    "stan/src/stan/lang/grammars/expression_grammar_inst.cpp",
    "stan/src/stan/lang/grammars/functions_grammar_inst.cpp",
    "stan/src/stan/lang/grammars/indexes_grammar_inst.cpp",
    "stan/src/stan/lang/grammars/program_grammar_inst.cpp",
    "stan/src/stan/lang/grammars/semantic_actions_def.cpp",
    "stan/src/stan/lang/grammars/statement_2_grammar_inst.cpp",
    "stan/src/stan/lang/grammars/statement_grammar_inst.cpp",
    "stan/src/stan/lang/grammars/term_grammar_inst.cpp",
    "stan/src/stan/lang/grammars/var_decls_grammar_inst.cpp",
    "stan/src/stan/lang/grammars/whitespace_grammar_inst.cpp",
    "stan2pyro/stan2pyro.cpp"
]
BUILD = "stan2pyro/build/"
names = list(map(lambda x: BUILD + ((x.split("/")[-1]).split(".")[0]) + ".o", sources))

BIN = "stan2pyro/bin/"



print("CMD=g++ %s -D FUSION_MAX_VECTOR_SIZE=12 -DBOOST_RESULT_OF_USE_TR1 -DBOOST_NO_DECLTYPE -DBOOST_DISABLE_ASSERTS -Os -ftemplate-depth-256 -Wno-unused-function -Wno-uninitialized -I stan/src -I stan/lib/stan_math/ -I stan/lib/stan_math/lib/eigen_3.3.3 -I stan/lib/stan_math/lib/boost_1.65.1 -I stan/lib/stan_math/lib/cvodes_2.9.0/include\n" % (DBG_OPTS))


print("\nall: %sstan2pyro" % BIN)

print("\nclean:")
print("\t-rm -rf %s %s" % (BIN,BUILD)) 

print("\n%sstan2pyro: %s %s" % (BIN, " ".join(names),BUILD))
print("\t$(CMD) -o %sstan2pyro %s*.o" % (BIN,BUILD))

print("\nexe: %sstan2pyro.o" % BUILD) 
print("\t$(CMD) -o %sstan2pyro %s*.o" % (BIN, BUILD)) 

for i in range(len(names)):
    print("\n%s: %s %s" % (names[i], sources[i], BUILD))
    print("\t$(CMD) -c -o %s %s" % ((names[i], sources[i])))
    
print("\n%s:"% BUILD)
print("\tmkdir -p %s" % BUILD)
print("\tmkdir -p %s" % BIN)

