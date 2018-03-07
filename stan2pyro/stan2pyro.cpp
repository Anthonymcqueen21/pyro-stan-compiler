#include <stan/version.hpp>
#include <stan/lang/compiler.hpp>
#include <stan/lang/ast.hpp>
#include <gen_pyro_statement.hpp>
#include <gen_pyro_expression.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/ast/node/var_decl.hpp>
#include <stan/lang/generator/has_lb.hpp>
#include <stan/lang/generator/has_lub.hpp>
#include <stan/lang/generator/has_ub.hpp>

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


    struct pyro_init_visgen : public visgen {
      size_t indent_;
      explicit pyro_init_visgen (size_t indent, std::ostream& o)
        : visgen(o), indent_(indent) {  }

      template <typename D>
      std::string function_args(const D& x) const {
        std::stringstream ss;
        if (has_lub(x)) {
          pyro_generate_expression(x.range_.low_.expr_, NOT_USER_FACING, ss);
          ss << get_dims_expand(x.dims_) << ",";
          pyro_generate_expression(x.range_.high_.expr_, NOT_USER_FACING, ss);
          ss << get_dims_expand(x.dims_);
        } else if (has_lb(x)) {
          pyro_generate_expression(x.range_.low_.expr_, NOT_USER_FACING, ss);
          ss << get_dims_expand(x.dims_);
          ss<<", 1. +";
          pyro_generate_expression(x.range_.low_.expr_, NOT_USER_FACING, ss);
          ss << get_dims_expand(x.dims_);
        } else if (has_ub(x)) {
          ss<<"-1. +";
          pyro_generate_expression(x.range_.low_.expr_, NOT_USER_FACING, ss);
          ss << get_dims_expand(x.dims_);
          ss<<",";
          pyro_generate_expression(x.range_.low_.expr_, NOT_USER_FACING, ss);
          ss << get_dims_expand(x.dims_);
        } else {
          ss << "to_variable(-2.)"<<get_dims_expand(x.dims_);
          ss<<",";
          ss<<" to_variable(2.)"<<get_dims_expand(x.dims_);
        }
        return ss.str();
      }

      std::string get_dims_expand(const std::vector<expression>& dims) const {
        std::stringstream ss;
        ss<<"(";
        int n_dims = dims.size();
        if (n_dims==0){
            return "";
        }
        for (int i=0;i<n_dims;i++){
            pyro_generate_expression(dims[i], NOT_USER_FACING, ss);
            if (i !=n_dims-1)
                ss<< ", ";
        }
        ss<<")";
        return ".expand(" + ss.str() + ")";
      }

      void operator()(const double_var_decl& x) const {
        int n_dims = x.dims_.size();

        o_<<"dist.Uniform("<<function_args(x)<<") # real/double";
        o_<<std::endl;
      }
      void operator()(const nil& /*x*/) const { }  // dummy

      void operator()(const int_var_decl& x) const {
        assert (false);
      }

      void operator()(const vector_var_decl& x) const {
        assert (false);
      }

      void operator()(const row_vector_var_decl& x) const {
        assert (false);
      }

      void operator()(const matrix_var_decl& x) const {
        assert (false);
      }

      void operator()(const unit_vector_var_decl& x) const {
        assert (false);
      }

      void operator()(const simplex_var_decl& x) const {
        assert (false);
      }

      void operator()(const ordered_var_decl& x) const {
        assert (false);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        assert (false);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        assert (false);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        assert (false);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        assert (false);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        assert (false);
      }
    };


  }
}

void extract_data(const std::vector<stan::lang::var_decl> data,
                  const std::pair<std::vector<stan::lang::var_decl>,
                                  std::vector<stan::lang::statement> > derived_data,
                  int n_td) {
    for (int i = 0; i < data.size(); i++) {
        stan::lang::generate_indent(1, std::cout);
        std::cout << data[i].name() <<  " = "<< "data[\"" << data[i].name() << "\"];\n";
    }

    if (n_td > 0) {
        for(int j=0; j<n_td; j++){
            std::string var_name = derived_data.first[j].name();
            stan::lang::generate_indent(1, std::cout);
            std::cout << var_name << " = data[\"" << var_name << "\"];\n";
        }
    }
}


//TODO: write a visitor struct for statement_ similar to statement_visgen.hpp in /stan/lang/generator/
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
        std::cout << "\ndef transformed_data(data):" << "\n";
        for(int j=0; j<n_td; j++){
            std::string var_name = p.derived_data_decl_.first[j].name();
            stan::lang::generate_indent(1, std::cout);
            std::cout << var_name << " = ";
            // TODO: for each dimension expression, output that expression
            int n_dims = p.derived_data_decl_.first[j].dims().size();
            if (n_dims == 0) std::cout<< "0." <<std::endl;
            else std::cout<< "torch.zeros(";
            for(int kk=0; kk<n_dims; kk++){
                stan::lang::pyro_generate_expression(p.derived_data_decl_.first[j].dims()[kk], NOT_USER_FACING, std::cout);
                if (kk != n_dims-1) std::cout<<",";
                else std::cout<<")\n";
            }
            pyro_statement(p.derived_data_decl_.second[j], p, 1, std::cout);
            stan::lang::generate_indent(1, std::cout);
            std::cout << "data[\"" << var_name << "\"] = ";
            std::cout << var_name << "\n";
        }
    }
    std::cout << "\ndef init_params(data, params):" << "\n";
    extract_data(p.data_decl_, p.derived_data_decl_, n_td);
    for (int i = 0; i < p.parameter_decl_.size(); i++) {
        stan::lang::generate_indent(1, std::cout);
        std::cout << "params[\"" << p.parameter_decl_[i].name() << "\"] = ";
        stan::lang::var_decl x = p.parameter_decl_[i];
        // TODO: use init_visgen
        stan::lang::pyro_init_visgen  iv(0,std::cout);
        boost::apply_visitor(iv, p.parameter_decl_[i].decl_);

    }
    std::cout << "\ndef model(data, params):" << "\n";
    extract_data(p.data_decl_, p.derived_data_decl_, n_td);
    for (int i = 0; i < p.parameter_decl_.size(); i++) {
        stan::lang::generate_indent(1, std::cout);
        std::cout << p.parameter_decl_[i].name() <<  "= params[\"" << p.parameter_decl_[i].name() << "\"];\n";
    }
    stan::lang::pyro_statement(p.statement_, p, 1, std::cout);
}

int main(int argc, char *argv[]) {
    assert(argc == 2);
    std::string  model_fname = argv[1];
    //std::cout<<"model_file_name: "<<model_fname<<std::endl;
    std::ifstream fin(model_fname.c_str());
    std::string mname_ = "temp_model";
    std::stringstream out;
    stan::lang::program p;
    bool valid_model = stan::lang::compile_ast(&std::cerr,fin,out,p,mname_);
    //std::cout<<out.str()<<" ";
    //std::cout<<valid_model<<std::endl;
    printer(p);
    return 0;
}
