import type { City, FilterValues } from "../types";

export default function Filters({
  value,
  onChange,
  cities,
  showKind = true,
}: {
  value: FilterValues;
  onChange: (v: FilterValues) => void;
  cities: City[];
  showKind?: boolean;
}) {
  const selected = cities.filter((city) => !value.city || city.id === value.city);
  const bodies = [...new Set(selected.flatMap((city) => city.bodies))].sort();

  function change(name: keyof FilterValues, next: string) {
    const updated = { ...value, [name]: next || undefined };
    if (name === "city" && updated.body && !bodiesFor(updated.city).includes(updated.body)) {
      updated.body = undefined;
    }
    onChange(updated);
  }

  function bodiesFor(cityId?: string) {
    return [
      ...new Set(
        cities.filter((city) => !cityId || city.id === cityId).flatMap((city) => city.bodies),
      ),
    ];
  }

  return (
    <div className="filters" data-testid="filters">
      <select
        data-testid="filter-city"
        aria-label="City"
        value={value.city ?? ""}
        onChange={(event) => change("city", event.target.value)}
      >
        <option value="">All cities</option>
        {cities.map((city) => (
          <option key={city.id} value={city.id}>
            {city.name}
          </option>
        ))}
      </select>
      <select
        data-testid="filter-body"
        aria-label="Body"
        value={value.body ?? ""}
        onChange={(event) => change("body", event.target.value)}
      >
        <option value="">All bodies</option>
        {bodies.map((body) => (
          <option key={body} value={body}>
            {body}
          </option>
        ))}
      </select>
      <input
        type="date"
        data-testid="filter-date-from"
        aria-label="From date"
        value={value.dateFrom ?? ""}
        onChange={(event) => change("dateFrom", event.target.value)}
      />
      <input
        type="date"
        data-testid="filter-date-to"
        aria-label="To date"
        value={value.dateTo ?? ""}
        onChange={(event) => change("dateTo", event.target.value)}
      />
      {showKind && (
        <select
          data-testid="filter-kind"
          aria-label="Kind"
          value={value.kind ?? ""}
          onChange={(event) => change("kind", event.target.value)}
        >
          <option value="">All kinds</option>
          <option value="agenda">Agenda</option>
          <option value="minutes">Minutes</option>
          <option value="transcript">Transcript</option>
        </select>
      )}
    </div>
  );
}
