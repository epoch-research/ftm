/**
 * Converts an HTML string to a DOM element
 */
function html(str) {
  let parentTag = 'div';
  if (str.trim().startsWith('<tr>')) {
    parentTag = 'tbody';
  }

  let tmp = document.createElement(parentTag);
  tmp.innerHTML = str.trim();
  let node = tmp.firstChild;
  return node;
}

/**
 * Creates a DOM element
 * @param {String} tag
 * @param {Dict}   opts Element attributes (such as class, input type...)
 *   special attributes:
 *      innerHTML: sets the innerHTML of the element
 */
function el(tag, opts) {
  let element = document.createElement(tag);
  for (let attribute in opts) {
    if (attribute == 'innerHTML') {
      element.innerHTML = opts[attribute];
      continue;
    }

    element.setAttribute(attribute, opts[attribute]);
  }
  return element;
}

/**
 * Converts a selector into a DOM element if the argument is a string, or
 * returns the argument if it was already a DOM element.
 */
function nodify(nodeOrQuery) {
  if (typeof(nodeOrQuery) == 'string') {
    return document.querySelector(nodeOrQuery);
  }
  return nodeOrQuery;
}

function arrayMin(arr) {
  return Math.min(...arr);
}

function arrayMax(arr) {
  return Math.max(...arr);
}

function removeItemFromArray(arr, item) {
  let index = arr.indexOf(item);
  if (index >= 0) {
    arr.splice(index, 1);
  }
}

function swap(arr, i, j) {
  let tmp = arr[i];
  arr[i] = arr[j];
  arr[j] = tmp;
}

function clean_number(str) {
  str = str.replace(/\.([0-9]*[1-9])?0*/g, ".$1"); // remove right zeroes 
  str = str.replace('e+', 'e');
  str = str.replace('.e', 'e'); // remove the decimal point, if no decimals
  str = str.replace(/\.$/, ''); // remove the decimal point, if no decimals
  return str;
}

function standard_format(x) {
  let str;
  if (typeof x == 'undefined') {
    str = 'undefined';
  } else if (typeof x == 'number') {
    let sign = (x < 0) ? -1 : +1;
    if (sign < 0) x = -x;

    if (x > 100 || x < 1e-3) {
      str = x.toExponential(3);
    } else {
      str = x.toFixed(3);
    }

    str = clean_number(str);

    if (sign < 0) str = '-' + str;
  } else {
    str = x.toString();
  }
  return str;
}

