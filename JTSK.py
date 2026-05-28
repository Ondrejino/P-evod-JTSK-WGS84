from pyproj import Transformer

# --- POMOCNÉ FUNKCE PRO FORMÁTOVÁNÍ ---

def decimal_to_dms(decimal_degree, is_lat):
    """
    Převede desetinné stupně (DD) na Stupně, minuty, vteřiny (DMS).
    is_lat = True (Zeměpisná šířka), False (Zeměpisná délka)
    """
    if is_lat:
        direction = 'N' if decimal_degree >= 0 else 'S'
    else:
        direction = 'E' if decimal_degree >= 0 else 'W'
        
    abs_dd = abs(decimal_degree)
    degrees = int(abs_dd)
    minutes = int((abs_dd - degrees) * 60)
    seconds = (abs_dd - degrees - minutes/60) * 3600
    
    return f"{degrees}° {minutes}' {seconds:.5f}\" {direction}"

# --- HLAVNÍ TRANSFORMAČNÍ LOGIKA (3D) ---

# Inicializace transformátorů (always_xy=True zajišťuje pořadí [Délka/X, Šířka/Y, Výška/Z])
transformer_to_wgs = Transformer.from_crs("EPSG:5514", "EPSG:4326", always_xy=True)
transformer_to_jtsk = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)


# --- PŘÍKLAD 1: S-JTSK (3D) -> WGS84 (3D) ---
print("--- PŘEVOD S-JTSK -> WGS84 (3D) ---")
jtsk_x = -600000.00
jtsk_y = -1050000.00
jtsk_z = 249.00  # Nadmořská výška (např. v systému Bpv)

# Přidán třetí parametr (jtsk_z) a očekáváme třetí výstup (wgs_alt)
wgs_lon, wgs_lat, wgs_alt = transformer_to_wgs.transform(jtsk_x, jtsk_y, jtsk_z)

print(f"Vstupní S-JTSK: Y={jtsk_x}, X={jtsk_y}, Z={jtsk_z} m")
print(f"Výstup WGS84 (Desetinné): Lat={wgs_lat:.7f}, Lon={wgs_lon:.7f}, Výška={wgs_alt:.3f} m")
print(f"Výstup WGS84 (DMS): {decimal_to_dms(wgs_lat, is_lat=True)}, {decimal_to_dms(wgs_lon, is_lat=False)}\n")


# --- PŘÍKLAD 2: WGS84 (3D) -> S-JTSK (3D) ---
print("--- PŘEVOD WGS84 -> S-JTSK (3D) ---")
wgs_lat_input = 49.8248881
wgs_lon_input = 15.3025219
wgs_alt_input = 293.30  # Elipsoidická výška z GPS

# Přidán třetí parametr (wgs_alt_input)
jtsk_x_out, jtsk_y_out, jtsk_z_out = transformer_to_jtsk.transform(wgs_lon_input, wgs_lat_input, wgs_alt_input)

print(f"Vstupní WGS84: Lat={wgs_lat_input}, Lon={wgs_lon_input}, Výška={wgs_alt_input} m")
print(f"Výstup S-JTSK: Y={jtsk_x_out:.3f}, X={jtsk_y_out:.3f}, Z={jtsk_z_out:.3f} m")