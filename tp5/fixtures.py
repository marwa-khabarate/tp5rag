"""Génère les manuels et photos fictifs originaux du TP."""
from io import BytesIO

import pymupdf as fitz
from PIL import Image, ImageDraw

from tp5.settings import DATA

MANUALS = {
    "PX-200": [
        ("Voyant papier clignotant / P02", "papier",
         "Le pictogramme représentant deux feuilles et une flèche signale un bourrage papier (code P02). Éteignez l’imprimante, retirez doucement le papier coincé dans le sens de sortie et rechargez le bac avant de rallumer."),
        ("Voyant toner / T12", "toner",
         "La goutte noire indique un niveau de toner faible (code T12). Vous pouvez continuer à imprimer et préparer une cartouche compatible."),
        ("Connexion Wi-Fi / W14", "wifi",
         "Le symbole Wi-Fi barré indique une perte de connexion (code W14). Sélectionnez à nouveau le réseau 2,4 GHz dans Paramètres > Réseau, puis relancez l'impression."),
    ],
    "LX-40": [
        ("Voyant température / H01", "chaleur",
         "Le thermomètre orange indique une température élevée (code H01). Éteignez le vidéoprojecteur et laissez-le refroidir pendant 20 minutes avant de vérifier la ventilation."),
    ],
}


def draw_symbol(symbol: str) -> Image.Image:
    image = Image.new("RGB", (240, 160), "#f2f6fb")
    draw = ImageDraw.Draw(image)
    ink = "#17304b"
    if symbol == "papier":
        draw.rectangle((56, 50, 137, 133), outline=ink, width=6)
        draw.rectangle((73, 32, 154, 114), fill="#f2f6fb", outline=ink, width=6)
        draw.line((91, 73, 135, 73), fill=ink, width=7)
        draw.polygon(((132, 61), (149, 73), (132, 85)), fill=ink)
    elif symbol == "toner":
        draw.polygon(((118, 26), (75, 88), (161, 88)), fill=ink)
        draw.ellipse((75, 56, 161, 137), fill=ink)
    elif symbol == "wifi":
        for boundary in ((54, 39, 186, 146), (79, 60, 161, 142), (102, 83, 138, 131)):
            draw.arc(boundary, 218, 322, fill=ink, width=7)
        draw.ellipse((114, 113, 126, 125), fill=ink)
        draw.line((67, 34, 172, 128), fill="#a13420", width=7)
    else:
        draw.rounded_rectangle((108, 26, 132, 115), radius=12, outline=ink, width=6)
        draw.line((120, 55, 120, 114), fill="#a13420", width=8)
        draw.ellipse((99, 102, 141, 144), fill="#a13420", outline=ink, width=5)
    return image


def png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate() -> None:
    manuals = DATA / "manuels"
    photos = DATA / "photos"
    manuals.mkdir(parents=True, exist_ok=True)
    photos.mkdir(parents=True, exist_ok=True)
    for model, sections in MANUALS.items():
        with fitz.open() as pdf:
            for page_no, (heading, symbol, paragraph) in enumerate(sections, 1):
                page = pdf.new_page()
                page.draw_rect((0, 0, 595, 155), fill=(0.08, 0.16, 0.25))
                page.insert_text((55, 72), f"Documentation technique | {model}", fontsize=24, color=(1, 1, 1))
                page.insert_text((55, 110), "Manuel fictif pour le TP5", fontsize=12, color=(0.8, 0.9, 1))
                page.insert_text((55, 205), heading, fontsize=20)
                page.insert_textbox((55, 235, 540, 365), paragraph.replace("’", "'"), fontsize=13, lineheight=1.45)
                page.insert_image((55, 370, 295, 530), stream=png(draw_symbol(symbol)))
                page.insert_text((55, 555), f"Figure {page_no} - {heading}", fontsize=11)
                page.insert_text((55, 790), f"{model} / SAV pédagogique                                       {page_no}", fontsize=10)
            pdf.save(manuals / f"manuel_{model}.pdf")
    picture = Image.new("RGB", (408, 306), "#53677b")
    frame = ImageDraw.Draw(picture)
    frame.rounded_rectangle((24, 28, 384, 278), radius=24, fill="#1d2732")
    picture.paste(draw_symbol("papier").resize((301, 200)), (54, 53))
    picture.save(photos / "img_px200_01.png")


if __name__ == "__main__":
    generate()

