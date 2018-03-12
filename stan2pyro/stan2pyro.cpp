#include <stan/version.hpp>
#include <stan/lang/compiler.hpp>
#include <stan/lang/ast.hpp>
#include <gen_pyro_statement.hpp>
#include <gen_pyro_expression.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/ast/node/var_decl.hpp>
#include <stan/lang/ast/node/vector_var_decl.hpp>
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
      //if (ai_size <= (e_num_dims + 1) || !base_type.is_matrix_type()) {

        std::string curr_str = expr;
        if (isLHS) o << expr;

        for (size_t n = 0; n < ai_size; ++n) {
          //o << '[';
          std::stringstream expr_ix;
          pyro_generate_expression_as_index(indexes[n], user_facing, expr_ix);
          if (! isLHS){
              curr_str = "_index_select(" + curr_str + ", " + expr_ix.str() + " - 1) ";
          }
          else{
            o << "[" << expr_ix.str() << " - 1]";
          }
          //o << " - 1]";
        }
        if (!isLHS) o << curr_str;
      //}
      /*else {
        //std::cout<<"generate_pyro_indexed_expr cannot be computed in this case ai_size="<<ai_size;
        //std::cout<<" e_num_dims="<<e_num_dims << "\n";
        //assert(false); //"generate_pyro_indexed_expr cannot be computed in this case");
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
      }*/
    }

    std::string get_dims(const std::vector<expression>& dims)  {
        std::stringstream ss;
        //ss<<"(";
        int n_dims = dims.size();
        if (n_dims==0){
            return "";
        }
        for (int i=0;i<n_dims;i++){
            pyro_generate_expression_as_index(dims[i], NOT_USER_FACING, ss);
            if (i !=n_dims-1)
                ss<< ", ";
        }
        //ss<<")";
        return ss.str();
        //return ", dims=(" +ss.str() + ")";
    }

    struct pyro_init_visgen : public visgen {
      size_t indent_;
      std::string var_name_;
      bool use_cache_;
      explicit pyro_init_visgen (size_t indent, std::ostream& o, std::string var_name, bool use_cache=true)
        : visgen(o), indent_(indent), var_name_(var_name), use_cache_(use_cache) {  }

      template <typename D>
      std::string function_args(const D& x) const {
        std::stringstream ss;
        if (has_lub(x)) {
          ss<<", low=";
          pyro_generate_expression_as_index(x.range_.low_.expr_, NOT_USER_FACING, ss);
          ss << ", high=";
          pyro_generate_expression_as_index(x.range_.high_.expr_, NOT_USER_FACING, ss);
        } else if (has_lb(x)) {
          ss<<", low=";
          pyro_generate_expression_as_index(x.range_.low_.expr_, NOT_USER_FACING, ss);
        } else if (has_ub(x)) {
          ss << ", high=";
          pyro_generate_expression_as_index(x.range_.high_.expr_, NOT_USER_FACING, ss);
        } else {
          ss<<"";
        }
        return ss.str();
      }



      void operator()(const double_var_decl& x) const {
        int n_dims = x.dims_.size();
        o_<<"init_real";
        if (use_cache_) o_<<"_and_cache";
        o_<<"(\""<< var_name_ <<"\""<<function_args(x);
        std::string str_dims = get_dims(x.dims_);
        if (str_dims != "") o_<<", dims=("<<str_dims <<")";
        o_ << ") # real/double";
        o_<<std::endl;
      }
      void operator()(const nil& /*x*/) const { }  // dummy

      void operator()(const int_var_decl& x) const {
        assert (false);
      }

      void operator()(const vector_var_decl& x) const {
        o_<<"init_vector";
        if (use_cache_) o_<<"_and_cache";
        o_<<"(\""<< var_name_ <<"\""<<function_args(x)<<", dims=(";
        std::string str_dims = get_dims(x.dims_);
        if (str_dims != "") o_<<str_dims<<", ";
        pyro_generate_expression_as_index(x.M_, NOT_USER_FACING, o_);
        o_<<")) # vector";
        o_<<std::endl;
      }

      void operator()(const row_vector_var_decl& x) const {
        assert (false);
      }

      void operator()(const matrix_var_decl& x) const {
        o_<<"init_matrix";
        if (use_cache_) o_<<"_and_cache";
        o_<<"(\""<< var_name_ <<"\""<<function_args(x)<<", dims=(";
        std::string str_dims = get_dims(x.dims_);
        if (str_dims != "") o_<<str_dims<<", ";
        pyro_generate_expression_as_index(x.M_, NOT_USER_FACING, o_);
        o_<<", ";
        pyro_generate_expression_as_index(x.N_, NOT_USER_FACING, o_);

        o_<<")) # matrix";
        o_<<std::endl;
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
        o_<<"init_matrix";
        if (use_cache_) o_<<"_and_cache";
        o_<<"(\""<< var_name_ <<"\""; //<<function_args(x);
        o_<<", dims=(";
        std::string str_dims = get_dims(x.dims_);
        if (str_dims != "") o_<<str_dims<<", ";
        pyro_generate_expression_as_index(x.K_, NOT_USER_FACING, o_);
        o_<<", ";
        pyro_generate_expression_as_index(x.K_, NOT_USER_FACING, o_);

        o_<<")) # cov-matrix";
        o_<<std::endl;
      }

      void operator()(const corr_matrix_var_decl& x) const {
        assert (false);
      }
    };


    struct pyro_varshape_visgen : public visgen {
      size_t indent_;
      std::string var_name_;

      explicit pyro_varshape_visgen (size_t indent, std::ostream& o, std::string var_name)
        : visgen(o), indent_(indent), var_name_(var_name){  }

      template <typename D>
      std::string function_args(const D& x) const {
        std::stringstream ss;
        if (has_lub(x)) {
          ss<<", low=";
          pyro_generate_expression_as_index(x.range_.low_.expr_, NOT_USER_FACING, ss);
          ss << ", high=";
          pyro_generate_expression_as_index(x.range_.high_.expr_, NOT_USER_FACING, ss);
        } else if (has_lb(x)) {
          ss<<", low=";
          pyro_generate_expression_as_index(x.range_.low_.expr_, NOT_USER_FACING, ss);
        } else if (has_ub(x)) {
          ss << ", high=";
          pyro_generate_expression_as_index(x.range_.high_.expr_, NOT_USER_FACING, ss);
        } else {
          ss<<"";
        }
        return ss.str();
      }



      void operator()(const double_var_decl& x) const {
        o_ <<"check_constraints(" <<var_name_<< function_args(x);
        std::string str_dims = get_dims(x.dims_);
        if (str_dims != "") o_<<", dims=["<<str_dims <<"]";
        else o_ <<", dims=[1]";
        o_<<")"<<std::endl;
      }
      void operator()(const nil& /*x*/) const { }  // dummy

      void operator()(const int_var_decl& x) const {
        o_ <<"check_constraints(" <<var_name_<< function_args(x);
        std::string str_dims = get_dims(x.dims_);
        if (str_dims != "") o_<<", dims=["<<str_dims <<"]";
        else o_ <<", dims=[1]";
        o_<<")"<<std::endl;
      }

      void operator()(const vector_var_decl& x) const {
        o_ <<"check_constraints(" <<var_name_<< function_args(x);
        std::string str_dims = get_dims(x.dims_);
        if (str_dims != "") o_<<", dims=["<<str_dims<<",";
        else o_ <<", dims=[";
        pyro_generate_expression_as_index(x.M_, NOT_USER_FACING, o_);
        o_<<"])"<<std::endl;;
      }

      void operator()(const row_vector_var_decl& x) const {
        assert (false);
      }

      void operator()(const matrix_var_decl& x) const {
        o_ <<"check_constraints(" <<var_name_<< function_args(x);
        std::string str_dims = get_dims(x.dims_);
        if (str_dims != "") o_<<", dims=["<<str_dims<<",";
        else o_ <<", dims=[";
        pyro_generate_expression_as_index(x.M_, NOT_USER_FACING, o_);
        o_<<", ";
        pyro_generate_expression_as_index(x.N_, NOT_USER_FACING, o_);
        o_<<"])"<<std::endl;;
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
        o_ <<"check_constraints(" <<var_name_; //<< function_args(x);
        std::string str_dims = get_dims(x.dims_);
        if (str_dims != "") o_<<", dims=["<<str_dims<<",";
        else o_ <<", dims=[";
        pyro_generate_expression_as_index(x.K_, NOT_USER_FACING, o_);
        o_<<", ";
        pyro_generate_expression_as_index(x.K_, NOT_USER_FACING, o_);
        o_<<"])"<<std::endl;;
      }

      void operator()(const corr_matrix_var_decl& x) const {
        assert (false);
      }
    };


    void generate_var_init_python(var_decl v, int indent, std::ostream& o){
        std::string var_name = v.name();
        generate_indent(indent, o);
        o << var_name << " = ";
        stan::lang::pyro_init_visgen  iv(0,o,var_name,false);
        boost::apply_visitor(iv, v.decl_);
        return;
        /*generate_indent(indent, o);
        o << var_name << " = ";

        if ( vector_var_decl* vec_v = boost::get<vector_var_decl>( &(v.decl_) ) ){
            // source:  http://www.boost.org/doc/libs/1_55_0/doc/html/variant/tutorial.html
            o<< "torch.zeros(";
            std::string str_dims = get_dims(vec_v->dims_);
            if (str_dims != "") o<<str_dims<<", ";
            stan::lang::pyro_generate_expression_as_index(vec_v->M_, NOT_USER_FACING, o);

            o<<")\n";
            return;
        }
        if ( matrix_var_decl* vec_v = boost::get<matrix_var_decl>( &(v.decl_) ) ){
            // source:  http://www.boost.org/doc/libs/1_55_0/doc/html/variant/tutorial.html
            o<< "torch.zeros(";
            std::string str_dims = get_dims(vec_v->dims_);
            if (str_dims != "") o<<str_dims<<", ";
            stan::lang::pyro_generate_expression_as_index(vec_v->M_, NOT_USER_FACING, o);
            o <<", ";
            stan::lang::pyro_generate_expression_as_index(vec_v->N_, NOT_USER_FACING, o);
            o<<")\n";
            return;
        }


        // TODO: for each dimension expression, output that expression
        int n_dims = v.dims().size();
        if (n_dims == 0) {
            if ( int_var_decl* iv = boost::get<int_var_decl>( &(v.decl_) ) ) o<<"0\n";
            else o<< "0.\n";
        }
        else o<< "torch.zeros(";
        for(int kk=0; kk<n_dims; kk++){
            pyro_generate_expression(v.dims()[kk], NOT_USER_FACING, o);
            if (kk != n_dims-1) o<<",";
            else o<<")\n";
        }*/
    }


    void generate_transformed_params_computation(const program &p, int indent, std::ostream& o){
        int n_td = p.derived_decl_.first.size();
        int n_td_s = p.derived_decl_.second.size();
        // assert(n_td == n_td_s);
        generate_indent(1, std::cout);
        std::cout<<"# INIT transformed parameters\n";
        for (int i=0;i < n_td; i++){
            var_decl vd = p.derived_decl_.first[i];
            generate_var_init_python(vd, indent, o);
        }
        for (int i=0;i < n_td_s; i++){
            //o << "# t-params i=" << i <<EOL;
            pyro_statement(p.derived_decl_.second[i], p, indent, o);
        }
    }

    void extract_data(const program &p, bool use_derived_data = true) {


        generate_indent(1, std::cout);
        std::cout<<"# INIT data\n";
        for (int i = 0; i < p.data_decl_.size(); i++) {
            generate_indent(1, std::cout);
            std::cout << p.data_decl_[i].name() <<  " = data[\"" << p.data_decl_[i].name() << "\"]\n";
        }

        int n_td = p.derived_data_decl_.first.size();

        if (n_td > 0 && use_derived_data) {
            stan::lang::generate_indent(1, std::cout);
            std::cout<<"# INIT transformed data\n";
            for(int j=0; j<n_td; j++){
                std::string var_name = p.derived_data_decl_.first[j].name();
                generate_indent(1, std::cout);
                std::cout << var_name << " = data[\"" << var_name << "\"]\n";
            }
        }
    }

  }
}






//TODO: write a visitor struct for statement_ similar to statement_visgen.hpp in /stan/lang/generator/
void printer(const stan::lang::program &p) {

    std::cout<<"def validate_data_def(data):"<<std::endl;
    int n_d = p.data_decl_.size();
    for(int j=0; j<n_d; j++){
        stan::lang::generate_indent(1, std::cout);
        std::string var_name = p.data_decl_[j].name();
        std::cout<<"assert '"<<var_name<<"' in data, 'variable not found in data: key="<<var_name<<"'"<<std::endl;
    }
    stan::lang::extract_data(p, false);

    std::stringstream ss_data_def; //to verify data dimensions / constraints in python
    for(int j=0; j<n_d; j++){
        std::string var_name = p.data_decl_[j].name();
        stan::lang::generate_indent(1, ss_data_def);
        //ss_data_def << "'"<<var_name<<"' : ";
        stan::lang::pyro_varshape_visgen  vv(0,ss_data_def, var_name);
        boost::apply_visitor(vv, p.data_decl_[j].decl_);
    }
    std::cout<<ss_data_def.str();

    int n_td = p.derived_data_decl_.first.size();
    int n_td_s = p.derived_data_decl_.second.size();

    if (n_td > 0) {

        std::cout << "\ndef transformed_data(data):" << "\n";
        stan::lang::extract_data(p, false);
        for(int j=0; j<n_td; j++){
            std::string var_name = p.derived_data_decl_.first[j].name();
            stan::lang::generate_var_init_python((p.derived_data_decl_.first[j]), 1, std::cout);
        }

        for(int j=0; j<n_td_s; j++){
            stan::lang::pyro_statement(p.derived_data_decl_.second[j], p, 1, std::cout);
        }
        for(int j=0; j<n_td; j++){
            std::string var_name = p.derived_data_decl_.first[j].name();
            stan::lang::generate_indent(1, std::cout);
            std::cout << "data[\"" << var_name << "\"] = ";
            std::cout << var_name << "\n";
        }

    }
    std::cout << "\ndef init_params(data, params):" << "\n";
    stan::lang::extract_data(p, true);

    stan::lang::generate_indent(1, std::cout);
    std::cout<<"# assign init values for parameters\n";
    for (int i = 0; i < p.parameter_decl_.size(); i++) {
        stan::lang::generate_indent(1, std::cout);
        std::string var_name = p.parameter_decl_[i].name();
        std::cout << "params[\"" << var_name << "\"] = ";
        stan::lang::var_decl x = p.parameter_decl_[i];
        stan::lang::pyro_init_visgen  iv(0,std::cout,var_name);
        boost::apply_visitor(iv, p.parameter_decl_[i].decl_);
    }
    std::cout << "\ndef model(data, params):" << "\n";
    stan::lang::extract_data(p, true);

    stan::lang::generate_indent(1, std::cout);
    std::cout<<"# INIT parameters\n";
    for (int i = 0; i < p.parameter_decl_.size(); i++) {
        stan::lang::generate_indent(1, std::cout);
        std::cout << p.parameter_decl_[i].name() <<  " = params[\"" << p.parameter_decl_[i].name() << "\"]\n";
    }
    stan::lang::generate_transformed_params_computation(p, 1, std::cout);

    stan::lang::generate_indent(1, std::cout);
    std::cout<<"# MODEL block"<<std::endl;
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
