/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "core/Fitness.h"

#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace topt {

struct FitnessExpr::Node {
  enum class Kind { Const, Var, Add, Sub, Mul, Div, Neg } kind;
  double value = 0.0;
  std::string var;
  std::shared_ptr<const Node> lhs, rhs;
};

namespace {

using NodePtr = std::shared_ptr<const FitnessExpr::Node>;

struct Parser {
  const std::string& text;
  std::size_t pos = 0;
  const std::vector<std::string>& vars;

  void skip_ws() {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
  }

  bool eat(char c) {
    skip_ws();
    if (pos < text.size() && text[pos] == c) {
      ++pos;
      return true;
    }
    return false;
  }

  [[noreturn]] void fail(const std::string& what) {
    throw std::invalid_argument("fitness expression: " + what + " at offset " +
                                std::to_string(pos) + " in '" + text + "'");
  }

  NodePtr make(FitnessExpr::Node&& n) {
    return std::make_shared<FitnessExpr::Node>(std::move(n));
  }

  NodePtr parse_expr() {
    NodePtr node = parse_term();
    for (;;) {
      if (eat('+')) {
        FitnessExpr::Node n;
        n.kind = FitnessExpr::Node::Kind::Add;
        n.lhs = node;
        n.rhs = parse_term();
        node = make(std::move(n));
      } else if (eat('-')) {
        FitnessExpr::Node n;
        n.kind = FitnessExpr::Node::Kind::Sub;
        n.lhs = node;
        n.rhs = parse_term();
        node = make(std::move(n));
      } else {
        return node;
      }
    }
  }

  NodePtr parse_term() {
    NodePtr node = parse_factor();
    for (;;) {
      if (eat('*')) {
        FitnessExpr::Node n;
        n.kind = FitnessExpr::Node::Kind::Mul;
        n.lhs = node;
        n.rhs = parse_factor();
        node = make(std::move(n));
      } else if (eat('/')) {
        FitnessExpr::Node n;
        n.kind = FitnessExpr::Node::Kind::Div;
        n.lhs = node;
        n.rhs = parse_factor();
        node = make(std::move(n));
      } else {
        return node;
      }
    }
  }

  NodePtr parse_factor() {
    skip_ws();
    if (pos >= text.size()) fail("unexpected end of input");
    char c = text[pos];
    if (c == '-') {
      ++pos;
      FitnessExpr::Node n;
      n.kind = FitnessExpr::Node::Kind::Neg;
      n.lhs = parse_factor();
      return make(std::move(n));
    }
    if (c == '(') {
      ++pos;
      NodePtr inner = parse_expr();
      if (!eat(')')) fail("missing ')'");
      return inner;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
      char* end = nullptr;
      double v = std::strtod(text.c_str() + pos, &end);
      if (end == text.c_str() + pos) fail("bad number");
      pos = static_cast<std::size_t>(end - text.c_str());
      FitnessExpr::Node n;
      n.kind = FitnessExpr::Node::Kind::Const;
      n.value = v;
      return make(std::move(n));
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      std::size_t start = pos;
      while (pos < text.size() &&
             (std::isalnum(static_cast<unsigned char>(text[pos])) || text[pos] == '_')) {
        ++pos;
      }
      std::string ident = text.substr(start, pos - start);
      std::string lowered;
      for (char ch : ident) lowered += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      for (const auto& known : vars) {
        if (lowered == known) {
          FitnessExpr::Node n;
          n.kind = FitnessExpr::Node::Kind::Var;
          n.var = lowered;
          return make(std::move(n));
        }
      }
      fail("unknown variable '" + ident + "'");
    }
    fail(std::string("unexpected character '") + c + "'");
  }
};

double eval_node(const FitnessExpr::Node& n, const std::map<std::string, double>& vars) {
  using Kind = FitnessExpr::Node::Kind;
  switch (n.kind) {
  case Kind::Const: return n.value;
  case Kind::Var: {
    auto it = vars.find(n.var);
    return it == vars.end() ? 0.0 : it->second;
  }
  case Kind::Add: return eval_node(*n.lhs, vars) + eval_node(*n.rhs, vars);
  case Kind::Sub: return eval_node(*n.lhs, vars) - eval_node(*n.rhs, vars);
  case Kind::Mul: return eval_node(*n.lhs, vars) * eval_node(*n.rhs, vars);
  case Kind::Div: {
    double denom = eval_node(*n.rhs, vars);
    return denom == 0.0 ? 0.0 : eval_node(*n.lhs, vars) / denom;
  }
  case Kind::Neg: return -eval_node(*n.lhs, vars);
  }
  return 0.0;
}

} // namespace

FitnessExpr FitnessExpr::parse(const std::string& text,
                               const std::vector<std::string>& knownVars) {
  Parser p{text, 0, knownVars};
  FitnessExpr expr;
  expr.root_ = p.parse_expr();
  p.skip_ws();
  if (p.pos != text.size()) p.fail("trailing characters");
  expr.text_ = text;
  return expr;
}

double FitnessExpr::eval(const std::map<std::string, double>& vars) const {
  if (!root_) return 0.0;
  return eval_node(*root_, vars);
}

} // namespace topt
