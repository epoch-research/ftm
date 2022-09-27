/*******************************************************************************

Poor people's implementation of some of NumPy's utilities

*******************************************************************************/

let nj;

{
  let exports = {
    nan: NaN,

    array: (array_or_count) => {
      if (Array.isArray(array_or_count)) return array_or_count;
      return Array(array_or_count);
    },

    zeros: (n) => new Array(n).fill(0),

    arange: (start, stop, step) => {
      let array = [];
      for (let x = start; x < stop; x += step) {
        array.push(x);
      }
      return array;
    },

    linspace: (a, b, count) => {
      let array = [];

      let step = (b - a)/(count - 1);
      for (let i = 0; i < count; i++) {
        array.push(a + i*step);
      }

      return array;
    },

    insert: (a, i, x) => a.splice(i, 0, x),

    concatenate: (a, b) => a.concat(b),

    min: (a) => Math.min(...a),

    max: (a) => Math.max(...a),

    maximum: (m, a) => exports.unaop(a, x => Math.max(x, m)),

    minimum: (m, a) => exports.unaop(a, x => Math.min(x, m)),

    clip: (m, a) => exports.unaop(a, x => Math.max(x, m)),

    argmax: (arr) => {
      let max = -Infinity;
      let imax = 0;
      for (let i = 0; i < arr.length; i++) {
        if (arr[i] > max) {
          imax = i;
          max = arr[i];
        }
      }

      return imax;
    },

    round: (x) => Math.round(x),

    even_round: (x) => {
      // This is Python's way of rounding (see https://docs.python.org/3/library/functions.html#round)
      let n = Math.round(x);
      let r = x % 1;
      if (Math.abs(r) == 0.5 && (n % 2) != 0) {
        n--;
      }
      return n;
    },

    log10: (a) => exports.unaop(a, Math.log10),

    exp: (a) => exports.unaop(a, Math.exp),

    sum: (a) => {
      let acc = 0;
      for (let x of a) acc += x;
      return acc;
    },

    mean: (a) => exports.sum(a)/a.length,

    all_equals: (arr, x) => {
      return arr.every(y => y == x);
    },

    any: (arr) => {
      for (let x of arr) if (x) return true;
      return false;
    },

    count_true: (a) => {
      let count = 0;
      for (let x of a) if (x) count++;
      return count;
    },

    /*
    // TODO If this is fast enough, use this code instead of the one below
    add:     (a, b) => exports.binop(a, b, (x, y) => x + y),
    sub:     (a, b) => exports.binop(a, b, (x, y) => x - y),
    mult:    (a, b) => exports.binop(a, b, (x, y) => x * y),
    div:     (a, b) => exports.binop(a, b, (x, y) => x / y),
    pow:     (a, b) => exports.binop(a, b, (x, y) => (x == 0) ? 0 : x ** y),
    lt:      (a, b) => exports.binop(a, b, (x, y) => x < y),
    gt:      (a, b) => exports.binop(a, b, (x, y) => x > y),
    gte:     (a, b) => exports.binop(a, b, (x, y) => x >= y),
    lte:     (a, b) => exports.binop(a, b, (x, y) => x <= y),
    and:     (a, b) => exports.binop(a, b, (x, y) => x && y),

    binop: (a, b, op) => {
      // Binary operation
      let c;

      if (Array.isArray(a) && Array.isArray(b)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = op(a[i], b[i]);
      } else if (Array.isArray(a)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = op(a[i], b);
      } else if (Array.isArray(b)) {
        c = exports.array(b.length);
        for (let i = 0; i < c.length; i++) c[i] = op(a, b[i]);
      } else {
        c = op(a, b);
      }

      return c;
    },
    */

    add: (a, b) => {
      let c;

      if (Array.isArray(a) && Array.isArray(b)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] + b[i];
      } else if (Array.isArray(a)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] + b;
      } else if (Array.isArray(b)) {
        c = exports.array(b.length);
        for (let i = 0; i < c.length; i++) c[i] = a + b[i];
      } else {
        c = a + b;
      }

      return c;
    },

    sub: (a, b) => {
      let c;

      if (Array.isArray(a) && Array.isArray(b)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] - b[i];
      } else if (Array.isArray(a)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] - b;
      } else if (Array.isArray(b)) {
        c = exports.array(b.length);
        for (let i = 0; i < c.length; i++) c[i] = a - b[i];
      } else {
        c = a - b;
      }

      return c;
    },

    mult: (a, b) => {
      let c;

      if (Array.isArray(a) && Array.isArray(b)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] * b[i];
      } else if (Array.isArray(a)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] * b;
      } else if (Array.isArray(b)) {
        c = exports.array(b.length);
        for (let i = 0; i < c.length; i++) c[i] = a * b[i];
      } else {
        c = a * b;
      }

      return c;
    },

    div: (a, b) => {
      let c;

      if (Array.isArray(a) && Array.isArray(b)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] / b[i];
      } else if (Array.isArray(a)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] / b;
      } else if (Array.isArray(b)) {
        c = exports.array(b.length);
        for (let i = 0; i < c.length; i++) c[i] = a / b[i];
      } else {
        c = a / b;
      }

      return c;
    },

    pow: (a, b) => {
      let c;

      if (Array.isArray(a) && Array.isArray(b)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = (a[i] == 0) ? 0 : a[i] ** b[i];
      } else if (Array.isArray(a)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = (a[i] == 0) ? 0 : a[i] ** b;
      } else if (Array.isArray(b)) {
        c = exports.array(b.length);
        for (let i = 0; i < c.length; i++) c[i] = (a == 0) ? 0 : a ** b[i];
      } else {
        c = (a == 0) ? 0 : a ** b;
      }

      return c;
    },

    lt: (a, b) => {
      let c;

      if (Array.isArray(a) && Array.isArray(b)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] < b[i];
      } else if (Array.isArray(a)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] < b;
      } else if (Array.isArray(b)) {
        c = exports.array(b.length);
        for (let i = 0; i < c.length; i++) c[i] = a < b[i];
      } else {
        c = a < b;
      }

      return c;
    },

    gt: (a, b) => {
      let c;

      if (Array.isArray(a) && Array.isArray(b)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] > b[i];
      } else if (Array.isArray(a)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] > b;
      } else if (Array.isArray(b)) {
        c = exports.array(b.length);
        for (let i = 0; i < c.length; i++) c[i] = a > b[i];
      } else {
        c = a > b;
      }

      return c;
    },

    gte: (a, b) => {
      let c;

      if (Array.isArray(a) && Array.isArray(b)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] >= b[i];
      } else if (Array.isArray(a)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] >= b;
      } else if (Array.isArray(b)) {
        c = exports.array(b.length);
        for (let i = 0; i < c.length; i++) c[i] = a >= b[i];
      } else {
        c = a >= b;
      }

      return c;
    },

    lte: (a, b) => {
      let c;

      if (Array.isArray(a) && Array.isArray(b)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] <= b[i];
      } else if (Array.isArray(a)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] <= b;
      } else if (Array.isArray(b)) {
        c = exports.array(b.length);
        for (let i = 0; i < c.length; i++) c[i] = a <= b[i];
      } else {
        c = a <= b;
      }

      return c;
    },

    and: (a, b) => {
      let c;

      if (Array.isArray(a) && Array.isArray(b)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] && b[i];
      } else if (Array.isArray(a)) {
        c = exports.array(a.length);
        for (let i = 0; i < c.length; i++) c[i] = a[i] && b;
      } else if (Array.isArray(b)) {
        c = exports.array(b.length);
        for (let i = 0; i < c.length; i++) c[i] = a && b[i];
      } else {
        c = a && b;
      }

      return c;
    },

    unaop: (a, op) => {
      // Unary operation
      return Array.isArray(a) ? a.map(x => op(x)) : op(a);
    },
  };

  nj = exports;
};
