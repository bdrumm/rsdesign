// Bundle entry: all Material Web components + typography + typescale CSS
import '@material/web/all.js';
import { styles as typescaleStyles } from '@material/web/typography/md-typescale-styles.js';
document.adoptedStyleSheets.push(typescaleStyles.styleSheet);
window.__mwcReady = true;
