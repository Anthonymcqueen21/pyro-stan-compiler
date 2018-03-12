#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/is_numbered_statement_vis.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>

#include <utility>
#include <vector>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>

#define NOT_USER_FACING false

namespace stan {
  namespace lang {

    void generate_var_init_python(var_decl v, int indent, std::ostream& o);

    void pyro_generate_expression(const expression& e, bool user_facing,
                                  std::ostream& o);

    std::string pyro_generate_expression_string(const expression& e,
                                                bool user_facing);

    void generate_idxs(const std::vector<idx>& idxs, std::ostream& o);

    void generate_statement(const std::vector<statement>& ss, int indent,
                            std::ostream& o);

    void pyro_statement(const statement& s, const program &p, int indent, std::ostream& o);

    template <bool isLHS>
    void generate_pyro_indexed_expr(const std::string& expr,
                               const std::vector<expression>& indexes,
                               base_expr_type base_type, size_t e_num_dims,
                               bool user_facing, std::ostream& o);

    void pyro_generate_expression_as_index(const expression& e, bool user_facing,
                             std::ostream& o);

    bool is_a_number(std::string s, double &n){
        try
        {
            n = boost::lexical_cast<double>(s);
            return true;
        }
        catch(boost::bad_lexical_cast &)
        {
            // if it throws, it's not a number.
            return false;
        }
    }

    /**
     * Visitor for generating statements.
     */
    struct pyro_statement_visgen : public visgen {
      /**
       * Indentation level.
       */
      size_t indent_;

      program p_;

      /**
       * Construct a visitor for generating statements at the
       * specified indent level to the specified stream.
       *
       * @param[in] indent indentation level
       * @param[in,out] o stream for generating
       */
      pyro_statement_visgen(size_t indent, std::ostream& o, const program& p)
        : visgen(o), indent_(indent) , p_(p) { }

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
          pyro_generate_expression(x.truncation_.high_.expr_, NOT_USER_FACING,
                              sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            pyro_generate_expression(x.dist_.args_[i], NOT_USER_FACING, sso_lp);
          }
          if (is_user_defined)
            sso_lp << ", pstream__";
          sso_lp << "), " << get_cdf(x.dist_.family_) << "(";
          pyro_generate_expression(x.truncation_.low_.expr_, NOT_USER_FACING,
                              sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            pyro_generate_expression(x.dist_.args_[i], NOT_USER_FACING, sso_lp);
          }
          if (is_user_defined)
            sso_lp << ", pstream__";
          sso_lp << "))";

        } else if (!x.truncation_.has_low() && x.truncation_.has_high()) {
          // T[,U];  -Dist_cdf_log(U)
          sso_lp << get_cdf(x.dist_.family_) << "(";
          pyro_generate_expression(x.truncation_.high_.expr_, NOT_USER_FACING,
                              sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            pyro_generate_expression(x.dist_.args_[i], NOT_USER_FACING, sso_lp);
          }
          if (is_user_defined)
            sso_lp << ", pstream__";
          sso_lp << ")";

        } else if (x.truncation_.has_low() && !x.truncation_.has_high()) {
          // T[L,]: -Dist_ccdf_log(L)
          sso_lp << get_ccdf(x.dist_.family_) << "(";
          pyro_generate_expression(x.truncation_.low_.expr_, NOT_USER_FACING,
                              sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            pyro_generate_expression(x.dist_.args_[i], NOT_USER_FACING, sso_lp);
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
          pyro_generate_expression(x.truncation_.low_.expr_, NOT_USER_FACING, o_);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            o_ << ", ";
            pyro_generate_expression(x.dist_.args_[i], NOT_USER_FACING, o_);
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
        // LHS
        std::stringstream ss_lhs;
        generate_indexed_expr<true>(x.var_dims_.name_,
                                    x.var_dims_.dims_,
                                    x.var_type_.base_type_,
                                    x.var_type_.dims_.size(),
                                    false,
                                    ss_lhs);
        std::string s_lhs = ss_lhs.str();
        o_ << s_lhs << " = _pyro_assign("<<s_lhs<< ", ";
        // RHS
        if (x.op_name_.size() == 0) {
          o_ << "(";
          generate_pyro_indexed_expr<false>(x.var_dims_.name_,
                                      x.var_dims_.dims_,
                                      x.var_type_.base_type_,
                                      x.var_type_.dims_.size(),
                                      false,
                                      o_);
          o_ << " " << x.op_ << " ";
          pyro_generate_expression(x.expr_, NOT_USER_FACING, o_);
          o_ << ")";
        } else {
          o_ << x.op_name_ << "(";
          generate_pyro_indexed_expr<false>(x.var_dims_.name_,
                                      x.var_dims_.dims_,
                                      x.var_type_.base_type_,
                                      x.var_type_.dims_.size(),
                                      false,
                                      o_);
          o_ << ", ";
          pyro_generate_expression(x.expr_, NOT_USER_FACING, o_);
          o_ << ")";
        }
        o_ << ")" << EOL;
      }

      void operator()(const assignment& x) const {
        // overwrite o_ to indicate ih for loop?
        generate_indent(indent_, o_);
        // LHS
        std::stringstream ss_lhs;
        generate_pyro_indexed_expr<true>(x.var_dims_.name_,
                                    x.var_dims_.dims_,
                                    x.var_type_.base_type_,
                                    x.var_type_.dims_.size(),
                                    false,
                                    ss_lhs);
        std::string s_lhs = ss_lhs.str();
        o_ << s_lhs << " = _pyro_assign("<<s_lhs<< ", ";
        // RHS
        // RHS
        if (x.var_dims_.dims_.size() == 0) {
            pyro_generate_expression(x.expr_, NOT_USER_FACING, o_);
        } else {
            //o_ << "to_float(";
            pyro_generate_expression(x.expr_, NOT_USER_FACING, o_);
            //o_ << ")";
        }
        o_ << ")" << EOL;
      }

      void operator()(const assgn& y) const {

        generate_indent(indent_, o_);
        o_ << "stan::model::assign(";

        expression var_expr(y.lhs_var_);
        pyro_generate_expression(var_expr, NOT_USER_FACING, o_);
        o_ << ", "
           << EOL;

        generate_indent(indent_ + 3, o_);
        generate_idxs(y.idxs_, o_);
        o_ << ", "
           << EOL;

        generate_indent(indent_ + 3, o_);
        if (y.lhs_var_occurs_on_rhs()) {
          o_ << "stan::model::deep_copy(";
          pyro_generate_expression(y.rhs_, NOT_USER_FACING, o_);
          o_ << ")";
        } else {
          pyro_generate_expression(y.rhs_, NOT_USER_FACING, o_);
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
        pyro_generate_expression(x, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
      }

      void generate_observe(const expression& e) const {
      // check if variable exists in data or it it is a constant
      // if so, generate observe statement
          std::string expr_str = pyro_generate_expression_string(e, NOT_USER_FACING);
          int n_d = p_.data_decl_.size();
          // iterate over  data block  and check if variable is in data
          for(int j=0;j<n_d; j++){
              if (expr_str == p_.data_decl_[j].name())
                  o_ << ", obs=" << expr_str;
          }
          // iterate over  data block  and check if variable is in transformed data
          int n_td = p_.derived_data_decl_.first.size();
          for(int j=0;j<n_td; j++){
              if (expr_str == p_.derived_data_decl_.first[j].name())
                  o_ << ", obs=" << expr_str;
          }

      }

      void operator()(const sample& x) const {
        std::string prob_fun = get_prob_fun(x.dist_.family_);

        generate_indent(indent_, o_);

        // since this is LHS -- using index based method makes sure that isLHS is set to True when calling
        // generate_expression for index_ops inside this
        std::stringstream ss;
        pyro_generate_expression_as_index(x.expr_, NOT_USER_FACING, ss);
        std::string lhs = ss.str();
        double n;
        bool is_num = is_a_number(lhs.c_str(), n);
        if (!is_num) {
            // conversion failed because the input wasn't a number
            o_ <<lhs << " = ";
        }
        else {
            // TODO: use this as observed value -- hack: output a temp variable with this const as its value
            // TODO: then use this as lhs / observe
            std::cerr << "SAMPLING CONSTANTS NOT SUPPORTED"<<std::endl;
            assert (false);
        }


        if ( const index_op* ix_op = boost::get<index_op>( &(x.expr_.expr_) ) ){
            // source:  http://www.boost.org/doc/libs/1_55_0/doc/html/variant/tutorial.html
            std::stringstream expr_o;
            pyro_generate_expression(ix_op->expr_, NOT_USER_FACING, expr_o);
            std::string expr_string = "\"" + expr_o.str();


            std::vector<std::string> indexes;
            for (size_t i = 0; i < ix_op->dimss_.size(); ++i){
              for (size_t j = 0; j < ix_op->dimss_[i].size(); ++j){
                std::stringstream ssi;
                pyro_generate_expression_as_index(ix_op->dimss_[i][j], NOT_USER_FACING, ssi);
                indexes.push_back(ssi.str());
                expr_string = expr_string + "[%d]";
              }
            }
            expr_string = expr_string + "\" % (";
            for (int ii=0; ii< indexes.size(); ii++){
                expr_string = expr_string + "to_int(" + indexes[ii] + "-1)";
                if(ii < indexes.size() - 1) expr_string = expr_string + ",";
                else expr_string = expr_string + ")";
            }
            lhs = expr_string;
        }
        else lhs = "\"" + lhs + "\"";

        o_ << " _pyro_sample(";
        //o_ << "lp_accum__.add(" << prob_fun << "<propto__>(";
        pyro_generate_expression(x.expr_, NOT_USER_FACING, o_);
        o_ << ", ";
        o_<<lhs;
        //pyro_generate_expression(x.expr_, NOT_USER_FACING, o_);
        o_<<", \"";
        std::string dist = x.dist_.family_;
        o_<<dist<<"\", [";
        for (size_t i = 0; i < x.dist_.args_.size(); ++i) {;
          if (i != 0) o_ << ", ";
          pyro_generate_expression(x.dist_.args_[i], NOT_USER_FACING, o_);
        }
        o_ << "]";
        generate_observe(x.expr_);
        o_ << ")" << EOL;

      }

      void operator()(const increment_log_prob_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "lp_accum__.add(";
        pyro_generate_expression(x.log_prob_, NOT_USER_FACING, o_);
        o_ << ");" << EOL;
      }

      void operator()(const statements& x) const {
        bool has_local_vars = x.local_decl_.size() > 0;
        if (has_local_vars) {
          generate_indent(indent_, o_);
          o_ << "# {" << EOL;
          for (int i=0; i < x.local_decl_.size(); i++){
            generate_var_init_python(x.local_decl_[i], indent_, o_);
          }
          //generate_local_var_decls(x.local_decl_, indent_, o_);
        }
        o_ << EOL;
        for (size_t i = 0; i < x.statements_.size(); ++i) {
          pyro_statement(x.statements_[i], p_, indent_, o_);
        }
        if (has_local_vars) {
          generate_indent(indent_, o_);
          o_ << "# }" << EOL;
        }
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
          pyro_generate_expression(rs.return_value_, NOT_USER_FACING, o_);
          o_ << ")";
        }
        o_ << EOL;
      }

      bool is_number(std::string& s) const {
          if (s[0] == '-') s = s.substr(1, s.size());
          std::string::const_iterator it = s.begin();
          while (it != s.end() && std::isdigit(*it)) ++it;
          return !s.empty() && it == s.end();
      }

      void operator()(const for_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "for " << x.variable_ << " in ";
        o_ << "range(";
        pyro_generate_expression_as_index(x.range_.low_, NOT_USER_FACING, o_);
        o_ <<", ";
        pyro_generate_expression_as_index(x.range_.high_, NOT_USER_FACING, o_);
        o_ <<" + 1):" << EOL;
        pyro_statement(x.statement_, p_, indent_ + 1, o_);
      }

      // TODO by example
      void operator()(const for_array_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "for (auto& " << x.variable_ << " : ";
        pyro_generate_expression(x.expression_, NOT_USER_FACING, o_);
        o_ << ") {" << EOL;
        generate_void_statement(x.variable_, indent_ + 1, o_);
        pyro_statement(x.statement_, p_, indent_ + 1, o_);
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const for_matrix_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "for (auto " << x.variable_ << "__loopid = ";
        pyro_generate_expression(x.expression_, NOT_USER_FACING, o_);
        o_ << ".data(); " << x.variable_ << "__loopid < ";
        pyro_generate_expression(x.expression_, NOT_USER_FACING, o_);
        o_ << ".data() + ";
        pyro_generate_expression(x.expression_, NOT_USER_FACING, o_);
        o_ << ".size(); ++" << x.variable_ << "__loopid) {" << EOL;
        generate_indent(indent_ + 1, o_);
        o_ << "auto& " << x.variable_ << " = *(";
        o_ << x.variable_ << "__loopid);"  << EOL;
        generate_void_statement(x.variable_, indent_ + 1, o_);
        pyro_statement(x.statement_, p_, indent_ + 1, o_);
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const while_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "while (as_bool(";
        pyro_generate_expression(x.condition_, NOT_USER_FACING, o_);
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
            o_ << " else: ";
          o_ << "if (as_bool(";
          pyro_generate_expression(x.conditions_[i], NOT_USER_FACING, o_);
          o_ << ")):" << EOL;
          pyro_statement(x.bodies_[i], p_, indent_ + 1, o_);
          generate_indent(indent_, o_);
          //o_ << '}';
        }
        if (x.bodies_.size() > x.conditions_.size()) {
          o_ << "else: " << EOL;
          pyro_statement(x.bodies_[x.bodies_.size()-1], p_, indent_ + 1, o_);
          generate_indent(indent_, o_);
          //o_ << '}';
        }
        o_ << EOL;
      }

      void operator()(const no_op_statement& /*x*/) const { }
    };

    void pyro_statement(const statement& s, const program &p, int indent, std::ostream& o) {

      if(false){
          is_numbered_statement_vis vis_is_numbered;
          if (boost::apply_visitor(vis_is_numbered, s.statement_)) {
            generate_indent(indent, o);
            o << "# current_statement_begin__ = " << s.begin_line_ << ";" << EOL;
          }
      }
      //std::cout<<"PYRO_STMT "<<s.begin_line_<<":"<<s.end_line_<<std::endl;
      pyro_statement_visgen vis(indent, o, p);
      boost::apply_visitor(vis, s.statement_);
    }

  }
}

