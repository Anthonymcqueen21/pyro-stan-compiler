#include <stan/version.hpp>
#include <stan/lang/compiler.hpp>
#include <stan/lang/ast.hpp>
#include <gen_pyro_statement.hpp>
#include <gen_pyro_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/ast/node/int_var_decl.hpp>

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

    template <bool isLHS>
    void generate_pyro_indexed_expr(const std::string& expr,
                               const std::vector<expression>& indexes,
                               base_expr_type base_type, size_t e_num_dims,
                               bool user_facing, std::ostream& o) {
      if (user_facing) {
        generate_indexed_expr_user(expr, indexes, o);
        return;
      }
      size_t ai_size = indexes.size();
      if (ai_size == 0) {
        o << expr;
        return;
      }
      if (ai_size <= (e_num_dims + 1) || !base_type.is_matrix_type()) {
        o << expr;
        for (size_t n = 0; n < ai_size; ++n) {
          o << '[';
          pyro_generate_expression_as_index(indexes[n], user_facing, o);
          o << ']';
        }
      } else {
        for (size_t n = 0; n < ai_size - 1; ++n)
          o << (isLHS ? "get_base1_lhs(" : "get_base1(");
        o << expr;
        for (size_t n = 0; n < ai_size - 2; ++n) {
          o << ',';
          pyro_generate_expression_as_index(indexes[n], user_facing, o);
          o << ',';
          generate_quoted_string(expr, o);
          o << ',' << (n+1) << ')';
        }
        o << ',';
        pyro_generate_expression_as_index(indexes[ai_size - 2U], user_facing, o);
        o << ',';
        pyro_generate_expression_as_index(indexes[ai_size - 1U], user_facing, o);
        o << ',';
        generate_quoted_string(expr, o);
        o << ',' << (ai_size - 1U) << ')';
      }
    }

  }
}



//TODO: write avisitor struct for statement_ similar to statement_visgen.hpp in /stan/lang/generator/
void printer(const stan::lang::program &p) {
    //int N = p.parameter_decl_.size(); //std::vector<var_decl>
    //std::cout<<"PRINTING PROGRAM: "<<N<<std::endl;
    //for(int i=0; i < N; i++){
    //    std::cout<<"hello"<<std::endl;
    //}
    /*int n = p.parameter_decl_.size();
    for (int i =0; i < n; i++){
        var_decl vd = p.parameter_decl_[i];
    }*/
    int n_td = p.derived_data_decl_.first.size();
    if (n_td > 0) {
        std::cout << "def transformed_data(data):" << "\n";
        for(int j=0; j<n_td; j++){
            std::string var_name = p.derived_data_decl_.first[j].name();
            pyro_statement(p.derived_data_decl_.second[j], p, 1, std::cout);
            stan::lang::generate_indent(1, std::cout);
            std::cout << "data[\"" << var_name << "\"] = ";
            std::cout << var_name << "\n";
        }
    }
    std::cout << "def init_params(params):" << "\n";
    for (int i = 0; i < p.parameter_decl_.size(); i++) {
        stan::lang::generate_indent(1, std::cout);
        std::cout << "params[\"" << p.parameter_decl_[i].name() << "\"] = ";
        std::cout << "dist.Uniform(to_var(0), to_var(10)) \n";
//         if (dynamic_cast<int_var_decl*>(&(p.parameter_decl_[i])) != NULL) {
//             int_var_decl* param = dynamic_cast<int_var_decl*>(&(p.parameter_decl_[i]));
//             if (param->has_low()) {
//                 pyro_generate_expression(param->low_, 1, std::cout);
//             if (param->has_high()) {
//                 pyro_generate_expression(param->high_, 1, std::cout);
//             }
//         }
    }
    std::cout << "def model(data):" << "\n";
    stan::lang::pyro_statement(p.statement_, p, 1, std::cout);
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
