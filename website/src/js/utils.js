/**
 * Converts an HTML string to a DOM element
 */
function html(str) {
  let tmp = document.createElement('div');
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
