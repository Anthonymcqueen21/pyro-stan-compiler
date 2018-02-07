#include <stan/version.hpp>
#include <stan/lang/compiler.hpp>
#include <stan/lang/ast.hpp>
#include <gen_pyro_statement.hpp>
#include <gen_pyro_expression.hpp>

#include <boost/variant/apply_visitor.hpp>
#include <ostream>

#include <utility>
#include <vector>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>

namespace stan {
  namespace lang {

    bool compile_ast(std::ostream* msgs, std::istream& in, std::ostream& out, program& prog,
                 const std::string& name, const bool allow_undefined = false,
                 const std::string& filename = "unknown file name",
                 const std::vector<std::string>& include_paths
                  = std::vector<std::string>()) {
      io::program_reader reader(in, filename, include_paths);
      std::string s = reader.program();
      std::stringstream ss(s);
      //program prog;
      bool parse_succeeded = parse(msgs, ss, name, reader, prog,
                                   allow_undefined);
      if (!parse_succeeded)
        return false;
      generate_cpp(prog, name, reader.history(), out);
      return true;
    }

  }
}



//TODO: write avisitor struct for statement_ similar to statement_visgen.hpp in /stan/lang/generator/
void printer(stan::lang::program &p) {
    std::cout<<"PRINTING PROGRAM"<<std::endl;
    //int N = p.parameter_decl_.size(); //std::vector<var_decl>
    //std::cout<<"PRINTING PROGRAM: "<<N<<std::endl;
    //for(int i=0; i < N; i++){
    //    std::cout<<"hello"<<std::endl;
    //}
    stan::lang::pyro_statement(p.statement_, 0, std::cout);
}

int main(int argc, char *argv[]) {
    assert(argc == 2);
    std::string  model_fname = argv[1];
    std::cout<<"model_file_name: "<<model_fname<<std::endl;
    std::ifstream fin(model_fname.c_str());
    std::string mname_ = "temp_model";
    std::stringstream out;
    stan::lang::program p;
    bool valid_model = stan::lang::compile_ast(&std::cerr,fin,out,p,mname_);
    //std::cout<<out.str()<<" ";
    std::cout<<valid_model<<std::endl;
    printer(p);
    return 0;
}
