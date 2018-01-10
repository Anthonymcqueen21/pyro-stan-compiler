#include <stan/version.hpp>
#include <stan/lang/compiler.hpp>
#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/is_numbered_statement_vis.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/statement_visgen.hpp>
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



namespace stan {
  namespace lang {

    void generate_idxs(const std::vector<idx>& idxs, std::ostream& o);

    void generate_statement(const std::vector<statement>& ss, int indent,
                            std::ostream& o);

    void pyro_statement(const statement& s, int indent, std::ostream& o);

    /**
     * Visitor for generating statements.
     */
    struct pyro_statement_visgen : public visgen {
      /**
       * Indentation level.
       */
      size_t indent_;

      /**
       * Construct a visitor for generating statements at the
       * specified indent level to the specified stream.
       *
       * @param[in] indent indentation level
       * @param[in,out] o stream for generating
       */
      pyro_statement_visgen(size_t indent, std::ostream& o)
        : visgen(o), indent_(indent) { }

      /**
       * Generate the target log density increments for truncating a
       * given density or mass function.
       *
       * @param[in] x sampling statement
       * @param[in] is_user_defined true if user-defined probability
       * function
       * @param[in] prob_fun name of probability function
       */
      void generate_truncation(const sample& x, bool is_user_defined,
                               const std::string& prob_fun) const {
        std::stringstream sso_lp;
        generate_indent(indent_, o_);
        if (x.truncation_.has_low() && x.truncation_.has_high()) {
          // T[L,U]: -log_diff_exp(Dist_cdf_log(U|params),
          //                       Dist_cdf_log(L|Params))
          sso_lp << "log_diff_exp(";
          sso_lp << get_cdf(x.dist_.family_) << "(";
          generate_expression(x.truncation_.high_.expr_, NOT_USER_FACING,
                              sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            generate_expression(x.dist_.args_[i], NOT_USER_FACING, sso_lp);
          }
          if (is_user_defined)
            sso_lp << ", pstream__";
          sso_lp << "), " << get_cdf(x.dist_.family_) << "(";
          generate_expression(x.truncation_.low_.expr_, NOT_USER_FACING,
                              sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            generate_expression(x.dist_.args_[i], NOT_USER_FACING, sso_lp);
          }
          if (is_user_defined)
            sso_lp << ", pstream__";
          sso_lp << "))";

        } else if (!x.truncation_.has_low() && x.truncation_.has_high()) {
          // T[,U];  -Dist_cdf_log(U)
          sso_lp << get_cdf(x.dist_.family_) << "(";
          generate_expression(x.truncation_.high_.expr_, NOT_USER_FACING,
                              sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            generate_expression(x.dist_.args_[i], NOT_USER_FACING, sso_lp);
          }
          if (is_user_defined)
            sso_lp << ", pstream__";
          sso_lp << ")";

        } else if (x.truncation_.has_low() && !x.truncation_.has_high()) {
          // T[L,]: -Dist_ccdf_log(L)
          sso_lp << get_ccdf(x.dist_.family_) << "(";
          generate_expression(x.truncation_.low_.expr_, NOT_USER_FACING,
                              sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            generate_expression(x.dist_.args_[i], NOT_USER_FACING, sso_lp);
          }
          if (is_user_defined)
            sso_lp << ", pstream__";
          sso_lp << ")";
        }

        o_ << "else lp_accum__.add(-";

        if (x.is_discrete() && x.truncation_.has_low()) {
          o_ << "log_sum_exp(" << sso_lp.str() << ", ";
          // generate adjustment for lower-bound off by 1 due to log CCDF
          o_ << prob_fun << "(";
          generate_expression(x.truncation_.low_.expr_, NOT_USER_FACING, o_);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            o_ << ", ";
            generate_expression(x.dist_.args_[i], NOT_USER_FACING, o_);
          }
          if (is_user_defined) o_ << ", pstream__";
          o_ << "))";
        } else {
          o_ << sso_lp.str();
        }

        o_ << ");" << std::endl;
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const compound_assignment& x) const {
        std::string op = boost::algorithm::erase_last_copy(x.op_, "=");
        generate_indent(indent_, o_);
        o_ << "stan::math::assign(";
        generate_indexed_expr<true>(x.var_dims_.name_,
                                    x.var_dims_.dims_,
                                    x.var_type_.base_type_,
                                    x.var_type_.dims_.size(),
                                    false,
                                    o_);
        o_ << ", ";
        if (x.op_name_.size() == 0) {
          o_ << "(";
          generate_indexed_expr<false>(x.var_dims_.name_,
                                      x.var_dims_.dims_,
                                      x.var_type_.base_type_,
                                      x.var_type_.dims_.size(),
                                      false,
                                      o_);
          o_ << " " << x.op_ << " ";
          generate_expression(x.expr_, NOT_USER_FACING, o_);
          o_ << ")";
        } else {
          o_ << x.op_name_ << "(";
          generate_indexed_expr<false>(x.var_dims_.name_,
                                      x.var_dims_.dims_,
                                      x.var_type_.base_type_,
                                      x.var_type_.dims_.size(),
                                      false,
                                      o_);
          o_ << ", ";
          generate_expression(x.expr_, NOT_USER_FACING, o_);
          o_ << ")";
        }
        o_ << ");" << EOL;
      }

      void operator()(const assignment& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::assign(";
        generate_indexed_expr<true>(x.var_dims_.name_,
                                    x.var_dims_.dims_,
                                    x.var_type_.base_type_,
                                    x.var_type_.dims_.size(),
                                    false,
                                    o_);
        o_ << ", ";
        generate_expression(x.expr_, NOT_USER_FACING, o_);
        o_ << ");" << EOL;
      }

      void operator()(const assgn& y) const {
        generate_indent(indent_, o_);
        o_ << "stan::model::assign(";

        expression var_expr(y.lhs_var_);
        generate_expression(var_expr, NOT_USER_FACING, o_);
        o_ << ", "
           << EOL;

        generate_indent(indent_ + 3, o_);
        generate_idxs(y.idxs_, o_);
        o_ << ", "
           << EOL;

        generate_indent(indent_ + 3, o_);
        if (y.lhs_var_occurs_on_rhs()) {
          o_ << "stan::model::deep_copy(";
          generate_expression(y.rhs_, NOT_USER_FACING, o_);
          o_ << ")";
        } else {
          generate_expression(y.rhs_, NOT_USER_FACING, o_);
        }

        o_ << ", "
           << EOL;
        generate_indent(indent_ + 3, o_);
        o_ << '"'
           << "assigning variable "
           << y.lhs_var_.name_
           << '"';
        o_ << ");"
           << EOL;
      }

      void operator()(const expression& x) const {
        generate_indent(indent_, o_);
        generate_expression(x, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
      }

      void operator()(const sample& x) const {
        std::string prob_fun = get_prob_fun(x.dist_.family_);
        generate_indent(indent_, o_);
        generate_expression(x.expr_, NOT_USER_FACING, o_);
        o_ << " = pyro.sample(\"";
        //o_ << "lp_accum__.add(" << prob_fun << "<propto__>(";
        generate_expression(x.expr_, NOT_USER_FACING, o_);
        o_<<"\","<<"dist."<<x.dist_.family_;
        for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
          o_ << ", ";
          generate_expression(x.dist_.args_[i], NOT_USER_FACING, o_);
        }
        o_ << ")" << EOL;
        
      }

      void operator()(const increment_log_prob_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "lp_accum__.add(";
        generate_expression(x.log_prob_, NOT_USER_FACING, o_);
        o_ << ");" << EOL;
      }

      void operator()(const statements& x) const {
        /*bool has_local_vars = x.local_decl_.size() > 0;
        if (has_local_vars) {
          generate_indent(indent_, o_);
          o_ << "{" << EOL;
          generate_local_var_decls(x.local_decl_, indent_, o_);
        }
        o_ << EOL;*/
        for (size_t i = 0; i < x.statements_.size(); ++i) {
          pyro_statement(x.statements_[i], indent_, o_);
        }
        /*if (has_local_vars) {
          generate_indent(indent_, o_);
          o_ << "}" << EOL;
        }*/
      }

      void operator()(const print_statement& ps) const {
        generate_indent(indent_, o_);
        o_ << "if (pstream__) {" << EOL;
        for (size_t i = 0; i < ps.printables_.size(); ++i) {
          generate_indent(indent_ + 1, o_);
          o_ << "stan_print(pstream__,";
          generate_printable(ps.printables_[i], o_);
          o_ << ");" << EOL;
        }
        generate_indent(indent_ + 1, o_);
        o_ << "*pstream__ << std::endl;" << EOL;
        generate_indent(indent_, o_);
        o_ << '}' << EOL;
      }

      void operator()(const reject_statement& ps) const {
        generate_indent(indent_, o_);
        o_ << "std::stringstream errmsg_stream__;" << EOL;
        for (size_t i = 0; i < ps.printables_.size(); ++i) {
          generate_indent(indent_, o_);
          o_ << "errmsg_stream__ << ";
          generate_printable(ps.printables_[i], o_);
          o_ << ";" << EOL;
        }
        generate_indent(indent_, o_);
        o_ << "throw std::domain_error(errmsg_stream__.str());" << EOL;
      }

      void operator()(const return_statement& rs) const {
        generate_indent(indent_, o_);
        o_ << "return ";
        if (!rs.return_value_.expression_type().is_ill_formed()
            && !rs.return_value_.expression_type().is_void()) {
          o_ << "stan::math::promote_scalar<fun_return_scalar_t__>(";
          generate_expression(rs.return_value_, NOT_USER_FACING, o_);
          o_ << ")";
        }
        o_ << ";" << EOL;
      }

      void operator()(const for_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "for (int " << x.variable_ << " = ";
        generate_expression(x.range_.low_, NOT_USER_FACING, o_);
        o_ << "; " << x.variable_ << " <= ";
        generate_expression(x.range_.high_, NOT_USER_FACING, o_);
        o_ << "; ++" << x.variable_ << ") {" << EOL;
        pyro_statement(x.statement_, indent_ + 1, o_);
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const for_array_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "for (auto& " << x.variable_ << " : ";
        generate_expression(x.expression_, NOT_USER_FACING, o_);
        o_ << ") {" << EOL;
        generate_void_statement(x.variable_, indent_ + 1, o_);
        pyro_statement(x.statement_, indent_ + 1, o_);
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const for_matrix_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "for (auto " << x.variable_ << "__loopid = ";
        generate_expression(x.expression_, NOT_USER_FACING, o_);
        o_ << ".data(); " << x.variable_ << "__loopid < ";
        generate_expression(x.expression_, NOT_USER_FACING, o_);
        o_ << ".data() + ";
        generate_expression(x.expression_, NOT_USER_FACING, o_);
        o_ << ".size(); ++" << x.variable_ << "__loopid) {" << EOL;
        generate_indent(indent_ + 1, o_);
        o_ << "auto& " << x.variable_ << " = *(";
        o_ << x.variable_ << "__loopid);"  << EOL;
        generate_void_statement(x.variable_, indent_ + 1, o_);
        pyro_statement(x.statement_, indent_ + 1, o_);
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const while_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "while (as_bool(";
        generate_expression(x.condition_, NOT_USER_FACING, o_);
        o_ << ")) {" << EOL;
        generate_statement(x.body_, indent_+1, o_);
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const break_continue_statement& st) const {
        generate_indent(indent_, o_);
        o_ << st.generate_ << ";" << EOL;
      }

      void operator()(const conditional_statement& x) const {
        for (size_t i = 0; i < x.conditions_.size(); ++i) {
          if (i == 0)
            generate_indent(indent_, o_);
          else
            o_ << " else ";
          o_ << "if (as_bool(";
          generate_expression(x.conditions_[i], NOT_USER_FACING, o_);
          o_ << ")) {" << EOL;
          pyro_statement(x.bodies_[i], indent_ + 1, o_);
          generate_indent(indent_, o_);
          o_ << '}';
        }
        if (x.bodies_.size() > x.conditions_.size()) {
          o_ << " else {" << EOL;
          pyro_statement(x.bodies_[x.bodies_.size()-1], indent_ + 1, o_);
          generate_indent(indent_, o_);
          o_ << '}';
        }
        o_ << EOL;
      }

      void operator()(const no_op_statement& /*x*/) const { }
    };

    void pyro_statement(const statement& s, int indent, std::ostream& o) {
      /*
      is_numbered_statement_vis vis_is_numbered;
      if (boost::apply_visitor(vis_is_numbered, s.statement_)) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " << s.begin_line_ << ";" << EOL;
      }*/
      
      pyro_statement_visgen vis(indent, o);
      boost::apply_visitor(vis, s.statement_);
    }

  }
}

//TODO: write avisitor struct for statement_ similar to statement_visgen.hpp in /stan/lang/generator/
void printer(stan::lang::program &p){
    std::cout<<"PRINTING PROGRAM"<<std::endl;
    //int N = p.parameter_decl_.size(); //std::vector<var_decl>
    //std::cout<<"PRINTING PROGRAM: "<<N<<std::endl;
    //for(int i=0; i < N; i++){
    //    std::cout<<"hello"<<std::endl;
    //}
    stan::lang::pyro_statement(p.statement_, 0, std::cout);
}

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
    //std::cout<<out.str()<<" ";
    std::cout<<valid_model<<std::endl;
    printer(p);
    return 0;
}