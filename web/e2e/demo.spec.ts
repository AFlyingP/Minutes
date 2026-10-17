import { expect, test } from "@playwright/test";

const NAV = [
  ["nav-search", "Search"],
  ["nav-facts", "Facts"],
  ["nav-ask", "Ask"],
  ["nav-agent", "Agent"],
  ["nav-label", "Label"],
];

test("demo scenario steps 2 to 7", async ({ page }) => {
  await page.goto("/");
  for (const [id, text] of NAV) await expect(page.getByTestId(id)).toHaveText(text);
  await expect(page.getByTestId("filter-city").locator("option")).toHaveText([
    "All cities",
    "Alder",
    "Birch",
  ]);
  await expect(page.getByTestId("filter-body")).toBeVisible();
  await expect(page.getByTestId("filter-date-from")).toBeVisible();
  await expect(page.getByTestId("filter-date-to")).toBeVisible();
  await expect(page.getByTestId("filter-kind").locator("option")).toHaveText([
    "All kinds",
    "Agenda",
    "Minutes",
    "Transcript",
  ]);

  const query = page.getByTestId("search-input");
  await query.fill("sidewalk repair contract");
  await page.getByTestId("filter-city").selectOption("birch");
  await query.press("Enter");
  const result = page.getByTestId("result-1");
  await expect(result.getByTestId("result-heading")).toHaveText(
    "Birch City Council, 2024-03-12, minutes, item 9.B",
  );
  await expect(result).toContainText("Granite Works Inc.");

  await result.getByTestId("result-span").getByText("p. 1").click();
  await expect(page).toHaveURL(/\/doc\/birch-minutes-201\?unit=1&start=\d+&end=\d+/);
  await expect(page.getByTestId("page-image")).toBeVisible();
  await expect(page.getByTestId("highlight").first()).toBeVisible();
  await expect(page.getByTestId("mark")).toBeVisible();

  await page.getByTestId("nav-ask").click();
  const ask = page.getByTestId("ask-input");
  await ask.fill("How did the council vote on the Granite Works contract in Resolution 2024-07?");
  await page.getByTestId("ask-submit").click();
  await expect(page.getByTestId("sentence")).toHaveText(
    "The roll call vote was 4 ayes and 1 noes.",
  );
  await expect(page.getByTestId("cite-1")).toHaveAttribute(
    "href",
    /^\/doc\/birch-minutes-201\?unit=2&start=\d+&end=\d+$/,
  );

  await ask.fill("What was decided about the airport runway extension?");
  await page.getByTestId("ask-submit").click();
  await expect(page.getByTestId("declined")).toHaveText(
    "I can't answer that from the indexed records.",
  );
  await expect(page.getByTestId("decline-reason")).toHaveText("no_retrieval");
});
