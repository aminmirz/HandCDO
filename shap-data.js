/* Rows of the paper's SHAP summary plot, top to bottom (sorted by mean |SHAP|).
   Ranges are the optimizer's search space (iros_paper_codebase/optimization/nested_optimizer_v2.py).
   Clips in media/params/ are rendered with generation_v2 over the same ranges (scripts/build_shap_parameters.py). */
window.HandCDOShap = (() => {
  const D = {
    F0: { name: 'Finger F0', swatch: '#2fc46a', where: 'right finger (green), beside the thumb' },
    F1: { name: 'Finger F1', swatch: '#e2c52e', where: 'middle finger (yellow)' },
    F2: { name: 'Finger F2', swatch: '#3b82f6', where: 'left finger (blue)' },
    T: { name: 'Thumb T', swatch: '#ef4f2c', where: 'thumb (red)' },
    K0: { name: 'Kernel K0', swatch: '#b3262e', where: 'first of the two palm surface kernels' },
    K1: { name: 'Kernel K1', swatch: '#b3262e', where: 'second of the two palm surface kernels' },
    all: { name: 'Whole hand', swatch: '#9a9aa0', where: 'shared by every finger and the thumb' },
    palm: { name: 'Palm surface', swatch: '#b3262e', where: 'both palm surface kernels' },
  };
  const M = {
    number: 'How many fingers the hand has besides the thumb. The optimizer chooses between three fingers and two (F2 removed).',
    normal: 'Pushes the digit’s mount outward from the palm edge, along the boundary normal.',
    side: 'Slides the digit’s mount sideways, parallel to the palm edge, while it keeps pointing the same way.',
    angle: 'Rotates the digit’s base in the palm plane, splaying it toward or away from its neighbours.',
    link: 'Extra length added to every link of the digit, in whole millimetres.',
    tipY: 'Scales every fingertip along its local Y axis (shared by all digits).',
    tipZ: 'Scales every fingertip along its local Z axis (shared by all digits).',
    fcode: 'The fingers’ joint-structure code: rotation mode plus the number and type of joints added before and after it. One code is shared by all fingers.',
    tcode: 'The thumb’s joint-structure code: rotation mode and its added joints.',
    deform: 'Peak height of the palm’s surface deformation. Both Gaussian kernels scale with it.',
    coff: 'How far the kernel’s centre sits from the palm centre, as a fraction of palm size.',
    cang: 'Direction of the kernel’s centre around the palm centre.',
    spread: 'Width (σ) of the Gaussian kernel relative to palm size: a sharp bump or a broad swell.',
    ratio: 'The kernel’s own height as a fraction of the max surface deform.',
  };
  const mm = (a, b) => `${a} to ${b} mm`;
  const rows = [
    ['finger-number', 'Finger Number', 'Structural', 'all', M.number, '2 or 3 fingers'],
    ['normal-offset-t', 'Normal Offset [T]', 'Normal Offset', 'T', M.normal, mm(-30, 30)],
    ['center-offset-k0', 'Center Offset [K₀]', 'Center Offset', 'K0', M.coff, '0 to 1'],
    ['link-length-f2', 'Added Link Length [F₂]', 'Link Length', 'F2', M.link, mm(0, 10)],
    ['center-angle-k1', 'Center Angle [K₁]', 'Center Angle', 'K1', M.cang, '0 to 360°'],
    ['side-offset-f2', 'Side Offset [F₂]', 'Side Offset', 'F2', M.side, mm(-30, 0)],
    ['side-offset-t', 'Side Offset [T]', 'Side Offset', 'T', M.side, mm(-40, 10)],
    ['link-length-f0', 'Added Link Length [F₀]', 'Link Length', 'F0', M.link, mm(0, 10)],
    ['angle-t', 'Angle [T]', 'Angle', 'T', M.angle, '−30 to 30°'],
    ['max-surface-deform', 'Max Surface Deform', 'Max Height', 'palm', M.deform, mm(0, 20)],
    ['fingertip-scale-z', 'Fingertip Scale Z', 'Fingertip Scale', 'all', M.tipZ, '0.5 to 1.5×'],
    ['center-offset-k1', 'Center Offset [K₁]', 'Center Offset', 'K1', M.coff, '0 to 1'],
    ['side-offset-f0', 'Side Offset [F₀]', 'Side Offset', 'F0', M.side, mm(0, 30)],
    ['normal-offset-f0', 'Normal Offset [F₀]', 'Normal Offset', 'F0', M.normal, mm(0, 5)],
    ['spread-k1', 'Spread [K₁]', 'Spread', 'K1', M.spread, '0.05 to 0.3'],
    ['angle-f2', 'Angle [F₂]', 'Angle', 'F2', M.angle, '−30 to 0°'],
    ['normal-offset-f1', 'Normal Offset [F₁]', 'Normal Offset', 'F1', M.normal, mm(0, 10)],
    ['intensity-ratio-k1', 'Intensity Ratio [K₁]', 'Intens. Ratio', 'K1', M.ratio, '0 to 1'],
    ['spread-k0', 'Spread [K₀]', 'Spread', 'K0', M.spread, '0.05 to 0.3'],
    ['angle-f0', 'Angle [F₀]', 'Angle', 'F0', M.angle, '0 to 30°'],
    ['fingertip-scale-y', 'Fingertip Scale Y', 'Fingertip Scale', 'all', M.tipY, '1.0 to 1.5×'],
    ['center-angle-k0', 'Center Angle [K₀]', 'Center Angle', 'K0', M.cang, '0 to 360°'],
    ['intensity-ratio-k0', 'Intensity Ratio [K₀]', 'Intens. Ratio', 'K0', M.ratio, '0 to 1'],
    ['normal-offset-f2', 'Normal Offset [F₂]', 'Normal Offset', 'F2', M.normal, mm(0, 5)],
    ['link-length-t', 'Added Link Length [T]', 'Link Length', 'T', M.link, mm(0, 10)],
    ['finger-code', 'Finger Code', 'Structural', 'all', M.fcode, '2 codes'],
    ['link-length-f1', 'Added Link Length [F₁]', 'Link Length', 'F1', M.link, mm(0, 10)],
    ['thumb-code', 'Thumb Code', 'Structural', 'T', M.tcode, '2 codes'],
  ].map(([id, label, group, digit, meaning, range], i) => ({ id, label, group, digit: D[digit], meaning, range, rank: i + 1 }));
  // Row centres in media/components/res-shap.webp, measured from the label column (fractions of image height).
  const geometry = { first: .0496, last: .8857, plotLeft: 0, plotRight: .885 };
  return { rows, geometry, digits: D };
})();
