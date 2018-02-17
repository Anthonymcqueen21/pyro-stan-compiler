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

namespace stan {
  namespace lang {


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
        generate_indexed_expr<true>(x.var_dims_.name_,
                                    x.var_dims_.dims_,
                                    x.var_type_.base_type_,
                                    x.var_type_.dims_.size(),
                                    false,
                                    o_);
        o_ << " = ";
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
        o_ << EOL;
      }

      void operator()(const assignment& x) const {
        generate_indent(indent_, o_);
        // LHS
        generate_pyro_indexed_expr<true>(x.var_dims_.name_,
                                    x.var_dims_.dims_,
                                    x.var_type_.base_type_,
                                    x.var_type_.dims_.size(),
                                    false,
                                    o_);
        o_ << " = ";
        // RHS
        pyro_generate_expression(x.expr_, NOT_USER_FACING, o_);
        o_ << EOL;
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
      // check if variable exists in data
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

        // PYRO_ADDED: identifying all transformed paramters expressions in the arguments of sampled
        // distributions and calling the relevant statement before using the variable
        // go over all argument expressions
        for (size_t i = 0; i < x.dist_.args_.size(); ++i) {;
          int n_tp = p_.derived_decl_.first.size();
          // compare each argument expression with variables in transformed parameters
          for(int j=0;j<n_tp; j++){
            std::string var_name = p_.derived_decl_.first[j].name();
            std::string expr_str = pyro_generate_expression_string(x.dist_.args_[i], NOT_USER_FACING);
            if (expr_str == var_name){
                // if they match generate the statement
                //o_<< indent_ <<

                pyro_statement(p_.derived_decl_.second[j], p_, indent_, o_);
                break;
            }
          }
          // pyro_generate_expression(x.dist_.args_[i], NOT_USER_FACING, o_);
        }

        generate_indent(indent_, o_);
        pyro_generate_expression(x.expr_, NOT_USER_FACING, o_);
        o_ << " = pyro.sample(\"";
        //o_ << "lp_accum__.add(" << prob_fun << "<propto__>(";
        pyro_generate_expression(x.expr_, NOT_USER_FACING, o_);
        std::string dist = x.dist_.family_;
        bool is_logit = false;
        std::vector<std::string> split;
        // capitalize first letter of distribution
        dist[0] = toupper(dist[0]);
        if (dist.find("logit") != std::string::npos) {
          // handles bernoulli and categorical logit distributions
            is_logit = true;
            std::string token;
            std::istringstream tokenStream(dist);
            while (std::getline(tokenStream, token, '_')) {
                split.push_back(token);
            }
            // only works for C++11
        //  boost::split(split, dist, [](char c){return c == '_';});
        }


        if (!is_logit) o_<<"\", "<<"dist."<<dist<<"(";
        else o_<<"\", "<<"dist."<<split[0]<<"(logit=";
        for (size_t i = 0; i < x.dist_.args_.size(); ++i) {;
          if (i != 0) o_ << ", ";
          pyro_generate_expression(x.dist_.args_[i], NOT_USER_FACING, o_);
        }
        o_ << ")";
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
        /*bool has_local_vars = x.local_decl_.size() > 0;
        if (has_local_vars) {
          generate_indent(indent_, o_);
          o_ << "{" << EOL;
          generate_local_var_decls(x.local_decl_, indent_, o_);
        }
        o_ << EOL;*/
        for (size_t i = 0; i < x.statements_.size(); ++i) {
          pyro_statement(x.statements_[i], p_, indent_, o_);
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
        std::stringstream tmp;
        o_ << "for " << x.variable_ << " in ";
        o_ << "range(";
        pyro_generate_expression_as_index(x.range_.low_, NOT_USER_FACING, tmp);
        std::string tmp_str = tmp.str();
        if (is_number(tmp_str)) {
            int index = atoi(tmp_str.c_str());
            o_ << index - 1 << ", ";
        } else
            o_ << tmp_str << " - 1, ";
        pyro_generate_expression_as_index(x.range_.high_, NOT_USER_FACING, o_);
        o_ <<"):" << EOL;
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
            o_ << " else ";
          o_ << "if (as_bool(";
          pyro_generate_expression(x.conditions_[i], NOT_USER_FACING, o_);
          o_ << ")) {" << EOL;
          pyro_statement(x.bodies_[i], p_, indent_ + 1, o_);
          generate_indent(indent_, o_);
          o_ << '}';
        }
        if (x.bodies_.size() > x.conditions_.size()) {
          o_ << " else {" << EOL;
          pyro_statement(x.bodies_[x.bodies_.size()-1], p_, indent_ + 1, o_);
          generate_indent(indent_, o_);
          o_ << '}';
        }
        o_ << EOL;
      }

      void operator()(const no_op_statement& /*x*/) const { }
    };

    void pyro_statement(const statement& s, const program &p, int indent, std::ostream& o) {
      /*
      is_numbered_statement_vis vis_is_numbered;
      if (boost::apply_visitor(vis_is_numbered, s.statement_)) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " << s.begin_line_ << ";" << EOL;
      }*/

      pyro_statement_visgen vis(indent, o, p);
      boost::apply_visitor(vis, s.statement_);
    }

  }
}

