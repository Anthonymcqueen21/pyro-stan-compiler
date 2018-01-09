#include <stan/version.hpp>
#include <stan/lang/compiler.hpp>
#include <stan/lang/ast.hpp>
/*
#include <stan/lang/ast/node/function_decl_def.hpp>
#include <stan/lang/ast/node/statement.hpp>
#include <stan/lang/ast/node/var_decl.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <stan/lang/ast/node/expression.hpp>
*/
#include <utility>
#include <vector>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>

/* REF: rstan/rstan/src/stanc.cpp */

typedef struct PyStancResult {
    int status;
    std::string msg; // for communicating errors
    std::string model_cppname;
    std::string cppcode;
} PyStancResult;
 

std::string stan_version() {
  std::string stan_version
    = stan::MAJOR_VERSION + "." +
      stan::MINOR_VERSION + "." +
      stan::PATCH_VERSION;
  return stan_version;
}

int stanc(std::string model_stancode, std::string model_name, PyStancResult& result) {
  static const int SUCCESS_RC = 0;
  static const int EXCEPTION_RC = -1;
  static const int PARSE_FAIL_RC = -2;

  /*
  std::string stan_version
    = stan::MAJOR_VERSION + "." +
      stan::MINOR_VERSION + "." +
      stan::PATCH_VERSION;
  */

  std::string mcode_ = model_stancode;
  std::string mname_ = model_name;

  std::stringstream out;
  std::istringstream in(mcode_);
  try {
    bool valid_model
      = stan::lang::compile(&std::cerr,in,out,mname_);
    if (!valid_model) {
      result.status = PARSE_FAIL_RC;
      return PARSE_FAIL_RC;
    }
  } catch(const std::exception& e) {
    result.status = EXCEPTION_RC;
    result.msg = e.what();
    return EXCEPTION_RC;
  }
  result.status = SUCCESS_RC;
  result.model_cppname = mname_;
  result.cppcode = out.str();
  return SUCCESS_RC;
}

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
void printer(stan::lang::program &p){
    std::cout<<"PRINTING PROGRAM"<<std::endl;
    int N = p.parameter_decl_.size(); //std::vector<var_decl>
    std::cout<<"PRINTING PROGRAM: "<<N<<std::endl;
    for(int i=0; i < N; i++){
        std::cout<<"hello"<<std::endl;
    }
}

//TODO: write a program printing function (recursively)
int main(int argc, char *argv[])
{
    assert(argc == 2);
    std::string  model_fname = argv[1];
    std::cout<<"model_file_name: "<<model_fname<<std::endl;
    std::ifstream fin(model_fname.c_str());
    std::string mname_ = "temp_model";
    std::stringstream out;
    stan::lang::program p;
    bool valid_model = stan::lang::compile_ast(&std::cerr,fin,out,p,mname_);
    std::cout<<out.str()<<" "<<valid_model<<std::endl;
    printer(p);
    return 0;
}
