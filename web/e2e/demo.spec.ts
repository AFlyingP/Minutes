import { expect, test } from "@playwright/test";

const NAV = [
  ["nav-search", "Search"],
  ["nav-facts", "Facts"],
  ["nav-ask", "Ask"],
  ["nav-agent", "Agent"],
  ["nav-label", "Label"],
];

test("demo scenario", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
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

  const runtime = globalThis as typeof globalThis & {
    process?: { env?: Record<string, string | undefined> };
  };
  if (runtime.process?.env?.MINUTES_SCREENSHOT === "1") {
    await page.screenshot({ path: "../docs/demo.png", fullPage: false });
  }

  await result.getByTestId("result-span").getByText("p. 1").click();
  await expect(page).toHaveURL(/\/doc\/birch-minutes-201\?unit=1&start=\d+&end=\d+/);
  await expect(page.getByTestId("page-image")).toBeVisible();
  await expect(page.getByTestId("highlight").first()).toBeVisible();
  await expect(page.getByTestId("mark")).toBeVisible();

  await page.getByTestId("nav-facts").click();
  await page.getByTestId("facts-kind").selectOption("votes");
  await page.getByTestId("filter-city").selectOption("birch");
  const facts = page.getByTestId("facts-table");
  await expect(facts.locator("tbody tr")).toHaveCount(4);
  const resolutionVote = facts.locator("tbody tr").filter({ hasText: "RESOLUTION 2024-07" });
  await expect(resolutionVote).toContainText("4");
  await expect(resolutionVote).toContainText("1");
  await expect(resolutionVote).toContainText("passed");
  await expect(resolutionVote.getByTestId("fact-source")).toHaveText("p. 2");

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

  await page.getByTestId("nav-agent").click();
  await page
    .getByTestId("agent-input")
    .fill("Which sidewalk repair contract items passed, and by what vote?");
  await page.getByTestId("agent-submit").click();
  await expect(page.getByTestId("step-1")).toContainText("1. search");
  await expect(page.getByTestId("step-2")).toContainText("2. sql");
  await expect(page.getByTestId("agent-answer")).toContainText("RESOLUTION 2024-07 passed 4-1");
  await expect(page.getByTestId("agent-status")).toHaveText("steps 2 of 8");
});
