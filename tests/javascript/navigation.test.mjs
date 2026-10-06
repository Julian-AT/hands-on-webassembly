import test from 'node:test';
import assert from 'node:assert/strict';
import { selection, selectionUrl } from '../../web/lib/navigation.mjs';
test('default and all seven assignment selections', () => {
  assert.deepEqual(
    [
      selection('https://course.test/').unit,
      selection('https://course.test/').normalize,
    ],
    [1, false],
  );
  for (let unit = 1; unit <= 7; unit++) {
    const got = selection(`https://course.test/?unit=${unit}`);
    assert.equal(got.unit, unit);
    assert.equal(got.normalize, false);
  }
});
test('invalid and duplicate selection normalizes and retains other values and fragment', () => {
  for (const query of [
    'unit=0',
    'unit=8',
    'unit=01',
    'unit=2.0',
    'unit=',
    'unit=hello',
    'unit=2&unit=2',
    'unit=2&unit=7',
  ]) {
    const got = selection(`https://course.test/?lang=en&${query}&x=a%20b#plot`);
    assert.equal(got.unit, 1);
    assert.equal(got.normalize, true);
    assert.deepEqual(got.url.searchParams.getAll('unit'), ['1']);
    assert.equal(got.url.searchParams.get('lang'), 'en');
    assert.equal(got.url.searchParams.get('x'), 'a b');
    assert.equal(got.url.hash, '#plot');
  }
});
test('selection URLs restore through history and reload without losing query or fragment', () => {
  const first = 'https://course.test/?lang=en#plot';
  const second = selectionUrl(first, 4).href;
  const third = selectionUrl(second, 7).href;
  assert.deepEqual(
    [first, second, third, second, first].map((url) => selection(url).unit),
    [1, 4, 7, 4, 1],
  );
  assert.equal(selection(third).unit, 7);
  assert.equal(new URL(third).searchParams.get('lang'), 'en');
  assert.equal(new URL(third).hash, '#plot');
  assert.throws(() => selectionUrl(first, 8));
  assert.throws(() => selectionUrl(first, 1.1));
});
