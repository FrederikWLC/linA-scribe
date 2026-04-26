import JSZip from 'jszip';
function sanitizeFileName(fileName) {
  return fileName
    .replace(/\.[^.]+$/, '')
    .replace(/[^a-zA-Z0-9-_\.\s]/g, '_')
    .trim() || 'export';
}

function getExportFileName(sourceName, extension) {
  const cleanName = sanitizeFileName(sourceName || 'export');
  const normalizedExtension = extension.startsWith('.') ? extension : `.${extension}`;
  return `${cleanName}${normalizedExtension}`;
}

function loadImage(imageUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Unable to load image for export.'));
    image.src = imageUrl;
  });
}

function makeWhiteTransparent(canvas) {
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Unable to create canvas context for export.');
  }

  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  // imagedata is apparently a one-dimensional array in RGBA order
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];

    if (a !== 0 && r === 255 && g === 255 && b === 255) {
      data[i + 3] = 0;
    }
  }

  context.putImageData(imageData, 0, 0);
}

function imageToBlob(image, mimeType, quality = 0.92) {
  const canvas = createCanvasFromImage(image, mimeType === 'image/png');

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error('Image export failed.'));
          return;
        }
        resolve(blob);
      },
      mimeType,
      mimeType === 'image/jpeg' ? 0.92 : undefined
    );
  });
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export async function exportImageAsPng(imageUrl, sourceName = 'export') {
  const image = await loadImage(imageUrl);
  const blob = await imageToBlob(image, 'image/png');
  const fileName = getExportFileName(sourceName, '.png');
  downloadBlob(blob, fileName);
  return blob;
}

export async function exportImageAsJpeg(imageUrl, sourceName = 'export') {
  const image = await loadImage(imageUrl);
  const blob = await imageToBlob(image, 'image/jpeg');
  const fileName = getExportFileName(sourceName, '.jpg');
  downloadBlob(blob, fileName);
  return blob;
}

function createCanvasFromImage(image, transparentWhite = false) {
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;

  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Unable to create canvas context for export.');
  }

  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  if (transparentWhite) {
    makeWhiteTransparent(canvas);
  }

  return canvas;
}

function makeThumbnailDataUrl(sourceDataUrl, width, height) {
  const image = new Image();
  image.src = sourceDataUrl;
  return new Promise((resolve, reject) => {
    image.onload = () => {
      const canvas = document.createElement('canvas');
      const scale = Math.min(256 / width, 256 / height, 1);
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Unable to create thumbnail canvas context.'));
        return;
      }
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL('image/png'));
    };
    image.onerror = () => reject(new Error('Unable to load image for ORA thumbnail.'));
  });
}

export async function exportImageAsOra(imageUrl, sourceName = 'export') {
  const image = await loadImage(imageUrl);
  const canvas = createCanvasFromImage(image, true);
  const mergedDataUrl = canvas.toDataURL('image/png');

  const zip = new JSZip();
  zip.file('mimetype', 'image/openraster', { compression: 'STORE' });

  const stackXml = `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<image w="${canvas.width}" h="${canvas.height}" version="0.0.1" xres="72" yres="72">\n` +
    `  <stack composite-op="svg:src-over" opacity="1" name="root" visibility="visible">\n` +
    `    <layer name="Layer 1" src="data/layer0.png" x="0" y="0" visibility="visible" opacity="1" composite-op="svg:src-over"/>\n` +
    `  </stack>\n` +
    `</image>`;

  zip.file('stack.xml', stackXml);
  zip.file('mergedimage.png', mergedDataUrl.split(',')[1], { base64: true });

  const thumbnailDataUrl = await makeThumbnailDataUrl(mergedDataUrl, canvas.width, canvas.height);
  zip.file('Thumbnails/thumbnail.png', thumbnailDataUrl.split(',')[1], { base64: true });
  zip.file('data/layer0.png', mergedDataUrl.split(',')[1], { base64: true });

  const blob = await zip.generateAsync({ type: 'blob' });
  const fileName = getExportFileName(sourceName, '.ora');
  downloadBlob(blob, fileName);
  return blob;
}

export async function exportImage(imageUrl, extension, sourceName = 'export') {
  if (!imageUrl) {
    throw new Error('No image URL provided for export.');
  }

  const normalizedExt = extension.toLowerCase().startsWith('.') ? extension.toLowerCase() : `.${extension.toLowerCase()}`;

  switch (normalizedExt) {
    case '.png':
      return exportImageAsPng(imageUrl, sourceName);
    case '.jpg':
    case '.jpeg':
      return exportImageAsJpeg(imageUrl, sourceName);
    case '.ora':
      return exportImageAsOra(imageUrl, sourceName);
    default:
      throw new Error(`Export for ${normalizedExt} is not implemented yet.`);
  }
}
