import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

st.set_page_config(page_title="ACO rutas", layout="wide")
st.title("ACO para TSP/VRP")

R = 6371.0 # Radio medio de la tierra en km
EPS = 1e-9
D_CACHE = {}

#Distancia ajustada para la tierra

def haversine_pair(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])  # convertimos grados a radianes
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2 # Para trabajar con trigonometricas
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))) # Distancia angular entre dos puntos de una esfera

# Distancia entre dos posiciones

def dist_ij(df, i, j):
    if i == j:
        return 0.0
    key = (int(i), int(j))
    if key in D_CACHE:
        return D_CACHE[key]
    d = haversine_pair(df.loc[i, "latitude"], df.loc[i, "longitude"],
                       df.loc[j, "latitude"], df.loc[j, "longitude"])
    D_CACHE[key] = d
    D_CACHE[(int(j), int(i))] = d # Guardamos la distanicia en ambas direcciones 
    # A -> B y B -> A
    return d

def costo_tour(tour, df):
    return float(sum(dist_ij(df, a, b) for a, b in zip(tour[:-1], tour[1:])))

def split_por_deposito(tour):
    rutas, cur = [], []
    for x in tour:
        cur.append(x)
        if x == 0 and len(cur) > 1:
            rutas.append(cur)
            cur = [0]
    if rutas and rutas[-1][-1] != 0:
        rutas[-1].append(0)
    return rutas

def evaporar(tau, rho):
    for k in list(tau.keys()):
        tau[k] *= (1 - rho)

def reforzar(tau, tour, df, Qtau):
    c = costo_tour(tour, df)
    dep = Qtau / max(c, EPS)
    for a, b in zip(tour[:-1], tour[1:]):
        tau[(a, b)] += dep
        tau[(b, a)] += dep

def factible(i, j, carga, demand, dist_ruta, df, Lmax=None):
    # Si carga es none estamos en tsp, sin capacidad, si hay capacidad comprobamos
    # que la demanda del cliente j que queremos visitar no exceda lo que queda en carga
    if (carga is not None) and (demand[j] > carga):
        return False
    # Si hay capacidad, comprobamos que la demanda del cliente j que quieres visitar
    # no exceda lo que queda de carga. Si excede es no factible
    if Lmax is not None:  # vuelta deposito
        if dist_ruta + dist_ij(df, i, j) + dist_ij(df, j, 0) > Lmax:
            return False
    return True

def sel_siguiente(i, remaining, carga, demand, dist_ruta, df, tau,
                  alpha=1.0, beta=3.0, q0=None, rng=None, Lmax=None):
    if rng is None:
        rng = np.random.default_rng()

    # Filtramos los destinos factibles según la capacidad y distancia
    F = [j for j in remaining if factible(i, j, carga, demand, dist_ruta, df, Lmax)]
    if not F:
        return None

    # Atractivo ACO: (feromona)^alpha * (1/distancia)^beta
    atractivos = []
    for j in F:
        d = max(dist_ij(df, i, j), EPS)  # EPS protege de divisiones por cero
        a = (tau[(i, j)] ** alpha) * ((1.0 / d) ** beta)
        atractivos.append(a)

    atractivos = np.array(atractivos, dtype=float)

    if q0 is not None and rng.random() < q0:
        idx_mejor = int(np.argmax(atractivos))
        return F[idx_mejor]

    # Cálculo de probabilidades normalizadas
    suma_atractivos = atractivos.sum()
    if suma_atractivos > 0:
        p = atractivos / suma_atractivos
    else:
        p = np.full(len(F), 1.0 / len(F))

    j_elegido = rng.choice(F, p=p)
    return j_elegido

def construir_hormiga(df, tau,
                      alpha=1.0, beta=3.0, q0=None, rng=None,
                      demand=None, Q=None, Lmax=None, Kmax=None):

    extra_permitido = False

    if rng is None:
        rng = np.random.default_rng()

    n = len(df)
    if demand is None:
        demand = np.zeros(n, dtype=float)

    pend = set(range(1, n))
    tour = [0]  # Iniciamos en el deposito
    nodo = 0
    cap = (Q if Q is not None else None)  # None = sin restriccion de capacidad
    dist_act = 0.0
    k = 1

    while pend:
        j = sel_siguiente(nodo, pend, cap, demand, dist_act, df, tau,
                          alpha=alpha, beta=beta, q0=q0, rng=rng, Lmax=Lmax)

        if j is not None:
            tour.append(j)
            pend.remove(j)
            if cap is not None:
                cap -= demand[j]
            dist_act += dist_ij(df, nodo, j)
            nodo = j
            continue

        if nodo != 0:
            tour.append(0)
            dist_act += dist_ij(df, nodo, 0)

        # Si definimos capacidad True y si k es igual k max true y si todavia hay pendientes en la lista True
        if (Kmax is not None) and (k >= Kmax) and pend:
            if not extra_permitido:
                Kmax += 1
                extra_permitido = True
            else:
                break

        nodo = 0
        dist_act = 0.0
        cap = (Q if Q is not None else None)
        k += 1

    if tour[-1] != 0:
        tour.append(0)
    return tour

with st.sidebar:
    up = st.file_uploader("Excel con columnas: latitude, longitude (opcional demand)",
                          type=["xlsx", "xls", "csv"])
    gen_dem = st.checkbox("Generar demanda si falta", True)
    dmin = st.number_input("Demanda mínima", 1, 10000, 5)
    dmax = st.number_input("Demanda máxima", 1, 10000, 20)

    st.markdown("---")
    alpha = st.slider("alpha (feromona)", 0.0, 5.0, 1.0, 0.1)
    beta = st.slider("beta (1/dist)", 0.0, 10.0, 3.0, 0.1)
    rho = st.slider("rho (evap.)", 0.0, 1.0, 0.5, 0.05)
    Qtau = st.number_input("Qtau", 0.0, 1000.0, 1.0, 0.1)
    q0 = st.slider("q0 (explotación)", 0.0, 1.0, 0.0, 0.05)

    st.markdown("---")
    iters = st.number_input("Iteraciones", 1, 100000, 50, 1)
    m = st.number_input("Hormigas/iteración", 1, 5000, 20, 1)
    seed = st.number_input("Semilla RNG (0=aleatoria)", 0, 999999, 123, 1)

    st.markdown("---")
    Qcap = st.number_input("Capacidad Q (0=TSP)", 0, 100000, 0, 1)
    Kmax = st.number_input("Máx. vehículos (0=sin tope)", 0, 1000, 3, 1)
    Lmax = st.number_input("Límite km por ruta (0=sin)", 0.0, 1e9, 0.0, 1.0)

run = st.button(" Ejecutar")

if run:
    if not up:
        st.error("Sube un archivo.")
        st.stop()

    if up.name.endswith(".csv"):
        df = pd.read_csv(up)
    else:
        df = pd.read_excel(up)
    df = df.rename(columns=str.lower).reset_index(drop=True)

    if {"latitud", "longitud"}.issubset(df.columns):
        df = df.rename(columns={"latitud": "latitude", "longitud": "longitude"})

    if not {"latitude", "longitude"}.issubset(df.columns):
        st.error("Faltan columnas latitude/longitude.")
        st.stop()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    if df[["latitude", "longitude"]].isna().any().any():
        st.error("Hay valores no numéricos en latitude/longitude.")
        st.stop()

    if "letras" not in df.columns:
        n = len(df)
        ABC = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if n <= 26:
            lab = ABC[:n]
        else:
            lab = ABC[:]
            k = 1
            while len(lab) < n:
                lab += [f"{c}_{k}" for c in ABC]
                k += 1
            lab = lab[:n]
        df.insert(0, "letras", lab)

    # demanda
    if "demand" in df.columns:
        df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0)

    if "demand" not in df.columns:
        if gen_dem:
            rngd = np.random.default_rng(None if seed == 0 else int(seed))
            df["demand"] = [0] + list(rngd.integers(dmin, dmax, size=len(df) - 1))
        else:
            df["demand"] = 0
    demand = df["demand"].to_numpy(float)

    tau0 = 1.0
    tau = defaultdict(lambda: tau0)
    rng = np.random.default_rng(None if seed == 0 else int(seed))

    Qarg = None if Qcap == 0 else float(Qcap)
    Karg = None if Kmax == 0 else int(Kmax)
    Larg = None if Lmax == 0 else float(Lmax)
    q0arg = None if q0 == 0 else float(q0)

    best_tour, best_cost = None, float("inf")
    hist = []
    D_CACHE.clear()

    prog = st.progress(0, text="Optimizando...")
    for it in range(int(iters)):
        sols = []
        for _ in range(int(m)):
            tour = construir_hormiga(
                df, tau,
                alpha=alpha, beta=beta, q0=q0arg, rng=rng,
                demand=demand, Q=Qarg, Lmax=Larg, Kmax=Karg
            )
            c = costo_tour(tour, df)
            sols.append((tour, c))

        evaporar(tau, rho)
        tour_it, cost_it = min(sols, key=lambda t: t[1])
        if cost_it < best_cost:
            best_tour, best_cost = tour_it, cost_it
        reforzar(tau, tour_it, df, Qtau)
        hist.append(best_cost)
        prog.progress((it + 1) / iters, text=f"Iteración {it + 1}/{int(iters)}")
    prog.empty()

    if best_tour is None:
        st.error("No se generó un tour (revisa Kmax/Qcap/Lmax).")
        st.stop()

    st.subheader(f"Mejor costo: {best_cost:.2f} km")
    st.write("Mejor tour (letras):", [df.loc[i, "letras"] for i in best_tour])

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(hist, lw=2)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Costo (km)")
    ax.set_title("Evolución del costo")
    ax.grid(True, alpha=.3)
    st.pyplot(fig)

    st.subheader("Rutas")
    rutas = split_por_deposito(best_tour)
    for k, r in enumerate(rutas, 1):
        letras = [df.loc[i, "letras"] for i in r]
        st.write(f"Vehículo {k}: {letras} | {costo_tour(r, df):.2f} km")

    fig2, ax2 = plt.subplots(figsize=(6.5, 5))
    ax2.scatter(df["longitude"], df["latitude"], s=35)
    for i, row in df.iterrows():
        ax2.text(row["longitude"], row["latitude"], f" {row['letras']}", fontsize=9)
    ax2.scatter([df.loc[0, "longitude"]], [df.loc[0, "latitude"]],
                s=140, marker="*", color="red", zorder=3)
    for r in rutas:
        xs = [df.loc[i, "longitude"] for i in r]
        ys = [df.loc[i, "latitude"] for i in r]
        ax2.plot(xs, ys, lw=2)
    ax2.set_xlabel("Longitud")
    ax2.set_ylabel("Latitud")
    ax2.grid(True, alpha=.3)
    st.pyplot(fig2)
