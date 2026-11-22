/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_CORE_FITNESS_H
#define TACUDA_OPT_CORE_FITNESS_H

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace topt {

// Compiled arithmetic expression over named metric variables.
// Grammar: expr := term (('+'|'-') term)*
//          term := factor (('*'|'/') factor)*
//          factor := number | identifier | function '(' expr (',' expr)* ')'
//                  | '-' factor | '(' expr ')'
//          function := abs | sqrt | log | min | max
// Identifiers are case-insensitive; see fitness_variable_names(). abs, sqrt
// and log (natural) take one argument, min and max two or more. A function
// applied outside its domain (sqrt(-1), log(0)) yields NaN/-inf, which the
// optimizer reports as an invalid variant rather than ranking it.
class FitnessExpr {
public:
  // Implementation detail, public only so the parser can build the tree.
  struct Node;

  // Throws std::invalid_argument on syntax errors or unknown identifiers.
  static FitnessExpr parse(const std::string& text,
                           const std::vector<std::string>& knownVars);

  double eval(const std::map<std::string, double>& vars) const;
  const std::string& text() const { return text_; }

private:
  std::shared_ptr<const Node> root_;
  std::string text_;
};

} // namespace topt

#endif // TACUDA_OPT_CORE_FITNESS_H
