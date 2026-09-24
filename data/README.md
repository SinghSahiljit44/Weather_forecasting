# Dati
```
data/
├── raw/        dati originali, sola lettura, mai modificati
├── interim/    serie oraria della sola area di Foggia (cache del passaggio lento)
└── processed/  dataset giornaliero finale, input dei modelli
```

## interim/

Serie **oraria** della sola area di Foggia, estratta dal `.nc`.

Esiste perché il file raw è in float64, senza compressione né chunking: leggere una cella
costa la scansione dell'intera variabile (~470 MB). Estrai una volta, poi riaggreghi a
giorni quante volte vuoi.

Cella di riferimento: **41,5 °N / 15,5 °E** (indici `lat=5`, `lon=8`), terraferma,
pressione media coerente con i 76 m di quota di Foggia. In alternativa la media delle 4
celle di pianura 41,4–41,5 °N × 15,5–15,6 °E. Da evitare la media sull'intero riquadro:
include il Gargano e i Monti Dauni, con circa 5 °C di differenza tra le celle.

## processed/

Dataset **giornaliero** finale, una riga per giorno, usato da tutti i modelli.

Colonne previste: `date`, `t2m_mean`, `t2m_min`, `t2m_max`, `tp_mm`, `e_mm`, `pev_mm`,
`skt_mean`, `stl1_mean`…`stl4_mean`, `swvl1_mean`…`swvl4_mean`.

Aggregazione: media (e min/max per `t2m`) sulle 24 ore per le variabili istantanee;
per quelle cumulate vedi la regola qui sotto.

## Da ricordare

1. **Variabili cumulate.** `tp`, `e` e `pev` si azzerano alle 01 UTC e crescono durante il
   giorno. Il totale del giorno D è quindi il valore alle **00:00 del giorno D+1**, non la
   somma delle 24 ore (che conterebbe tutto più volte). All'interno della giornata si
   vedono cali fino a 0,00001 mm: è rumore di arrotondamento, si ignora.
2. **Ultimo giorno.** Il 2025-12-31 non ha il proprio totale di pioggia, perché mancano le
   00:00 del 2026-01-01. Va scartato.
3. **Fuso orario.** I giorni sono in UTC, non in ora italiana.
