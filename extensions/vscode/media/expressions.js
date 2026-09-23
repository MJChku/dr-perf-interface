/* Small exact-integer language for user-proposed STATE relationships.
 * PCVs remain ordinary expressions in the annotated program's language.
 * This parser never executes JavaScript or the measured program.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.DrperfExpressions = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const precedence = {
    '||': 1,
    '&&': 2,
    '==': 3,
    '!=': 3,
    '<': 4,
    '<=': 4,
    '>': 4,
    '>=': 4,
    '+': 5,
    '-': 5,
    '*': 6,
    '//': 6,
    '%': 6,
    '**': 8
  };
  function compile(source) {
    if (typeof source !== 'string' || source.length > 4096)
      throw new Error('Relationship expression must be at most 4096 characters.');
    const tokens = [];
    const pattern =
      /\s*(?:(\d+)|("(?:[^"\\]|\\.)*")|([A-Za-z_][A-Za-z_0-9]*)|(\*\*|\/\/|<=|>=|==|!=|&&|\|\||[()+*%,?:!<>-]))/y;
    let offset = 0;
    while (source.slice(offset).trim()) {
      pattern.lastIndex = offset;
      const match = pattern.exec(source);
      if (!match) throw new Error(`Unexpected expression character at ${offset + 1}.`);
      offset = pattern.lastIndex;
      tokens.push({
        kind: match[1] ? 'integer' : match[2] ? 'string' : match[3] ? 'name' : 'op',
        text: match[1] || match[2] || match[3] || match[4]
      });
      if (tokens.length > 256)
        throw new Error('Relationship expression is too complex (256 tokens maximum).');
    }
    let position = 0,
      depth = 0;
    const features = new Map();
    const peek = () => tokens[position]?.text;
    const consume = (expected) => {
      const token = tokens[position++];
      if (!token || (expected && token.text !== expected))
        throw new Error(`Expected ${expected || 'expression'}.`);
      return token;
    };
    function expression(minimum = 0) {
      if (++depth > 48) throw new Error('Relationship expression is nested too deeply.');
      const token = consume();
      let left;
      if (token.kind === 'integer') {
        if (token.text.length > 128) throw new Error('Integer literal is too large.');
        left = { kind: 'integer', value: token.text };
      } else if (['+', '-', '!'].includes(token.text))
        left = { kind: 'unary', op: token.text, child: expression(7) };
      else if (token.text === '(') {
        left = expression();
        consume(')');
      } else if (token.kind === 'name' && ['last', 'cum', 'cumend', 'count'].includes(token.text)) {
        consume('(');
        const region = consume();
        if (region.kind !== 'string')
          throw new Error('History region names must be double-quoted strings.');
        let state = null;
        if (token.text !== 'count') {
          consume(',');
          const field = consume();
          if (field.kind !== 'string')
            throw new Error('History PCV names must be double-quoted strings.');
          state = JSON.parse(field.text);
        }
        consume(')');
        const feature = { kind: token.text, region: JSON.parse(region.text), state };
        const key = JSON.stringify(feature);
        features.set(key, feature);
        left = { kind: 'feature', feature };
      } else
        throw new Error(
          'Expected an integer, parenthesized expression, or last/cum/cumend/count history reference.'
        );
      while (Object.hasOwn(precedence, peek()) && precedence[peek()] >= minimum) {
        const op = consume().text,
          p = precedence[op];
        left = { kind: 'binary', op, left, right: expression(p + (op === '**' ? 0 : 1)) };
      }
      if (minimum === 0 && peek() === '?') {
        consume('?');
        const yes = expression();
        consume(':');
        left = { kind: 'conditional', condition: left, yes, no: expression() };
      }
      --depth;
      return left;
    }
    const tree = expression();
    if (position !== tokens.length) throw new Error('Unexpected trailing expression tokens.');
    return { source, tree, features: [...features.values()] };
  }
  function evaluate(compiled, featureValue) {
    let steps = 0;
    const bounded = (n) => {
      if (n.toString(2).length > 4096)
        throw new Error('Relationship arithmetic exceeds 4096 bits.');
      return n;
    };
    const floorDivide = (a, b) => {
      if (b === 0n) throw new Error('Division by zero in relationship.');
      const q = a / b,
        r = a % b;
      return r !== 0n && a < 0n !== b < 0n ? q - 1n : q;
    };
    function visit(node) {
      if (++steps > 256) throw new Error('Relationship evaluation budget exceeded.');
      if (node.kind === 'integer') return BigInt(node.value);
      if (node.kind === 'feature') return bounded(BigInt(featureValue(node.feature)));
      if (node.kind === 'unary') {
        const x = visit(node.child);
        return node.op === '-' ? -x : node.op === '!' ? (x === 0n ? 1n : 0n) : x;
      }
      if (node.kind === 'conditional')
        return visit(node.condition) !== 0n ? visit(node.yes) : visit(node.no);
      const a = visit(node.left);
      if (node.op === '&&') return a !== 0n && visit(node.right) !== 0n ? 1n : 0n;
      if (node.op === '||') return a !== 0n || visit(node.right) !== 0n ? 1n : 0n;
      const b = visit(node.right);
      switch (node.op) {
        case '+':
          return bounded(a + b);
        case '-':
          return bounded(a - b);
        case '*':
          return bounded(a * b);
        case '//':
          return floorDivide(a, b);
        case '%':
          return a - floorDivide(a, b) * b;
        case '**':
          if (b < 0n || b > 16n) throw new Error('Relationship exponent must be between 0 and 16.');
          return bounded(a ** b);
        case '<':
          return a < b ? 1n : 0n;
        case '<=':
          return a <= b ? 1n : 0n;
        case '>':
          return a > b ? 1n : 0n;
        case '>=':
          return a >= b ? 1n : 0n;
        case '==':
          return a === b ? 1n : 0n;
        case '!=':
          return a !== b ? 1n : 0n;
        default:
          throw new Error('Unsupported expression operation.');
      }
    }
    return visit(compiled.tree);
  }
  return { compile, evaluate };
});
