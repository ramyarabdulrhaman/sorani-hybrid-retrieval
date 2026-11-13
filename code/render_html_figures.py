"""Render the HTML figure sources to HD PNG (4x) and vector PDF using headless Edge/Chrome."""
import os, subprocess, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(BASE, 'code', 'figure_html')
OUT = os.path.join(BASE, 'figures')
BROWSERS = [r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Google\Chrome\Application\chrome.exe']
FIGS = {
    'fig1': ('fig1_kurdrag_architecture',     1260, 846),
        'fig10': ('fig10_morpho_bpe_tokenization_case_study', 1180, 640),
}
SCALE = 4


def render():
    browser = next(b for b in BROWSERS if os.path.exists(b))
    for src, (name, w, h) in FIGS.items():
        url = 'file:///' + os.path.join(HTML, src + '.html').replace('\\', '/')
        common = [browser, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-pdf-header-footer']
        subprocess.run(common + [f'--force-device-scale-factor={SCALE}', f'--window-size={w},{h}',
                                 f'--screenshot={os.path.join(OUT, name + ".png")}', url], check=True)
        subprocess.run(common + [f'--print-to-pdf={os.path.join(OUT, name + ".pdf")}', url], check=True)
        print('rendered', name)


if __name__ == '__main__':
    sys.exit(render())

