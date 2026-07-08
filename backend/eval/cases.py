"""
Text-only eval cases for the agent testing lab.

Each case has a realistic meeting transcript, ground-truth facts for coverage
scoring, and optional expected action owners/tasks for action recall scoring.

No external downloads — all transcripts are hand-authored for representativeness.
"""
from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    domain: str
    title: str
    source: str
    transcript: str
    ground_truth: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# General — baseline domain, catch-all conversations
# ---------------------------------------------------------------------------

_G1_TRANSCRIPT = """
Alice: Good morning everyone. Let's do a quick standup. I'll go first — yesterday I finished the new
user dashboard component. Blocked on backend API changes from Dan. Today I'll start integration tests.

Bob: I deployed the payment service to staging yesterday. It's passing all smoke tests. Today I'm
writing the load test harness. No blockers.

Carol: Still working on the mobile push notification flow. Hit a weird edge case where tokens expire
silently on iOS 17. Blocking issue — need someone from iOS team to look at it. I filed ticket MN-412.

Dan: I owe Alice the updated endpoint spec. I'll have that out by noon. Also I pushed the config
service changes to prod last night — monitor shows clean. No blockers from me.

Alice: Great. Carol — I'll ping Marcus from iOS team right after this call. Dan, once I have the spec
I can unblock myself.

Bob: One more thing — sprint demo is Friday at 2pm. Everyone should have their features in staging by
Thursday EOD.

Alice: Agreed. Let's wrap there. Thanks everyone.
"""

_G2_TRANSCRIPT = """
Marcus: Alright, Q3 budget review. Finance sent over the actuals this morning. We came in at
$840,000 against a budget of $900,000 — about 7% under. Infrastructure was the biggest savings:
we migrated three services to ARM instances and cut cloud spend by $22,000.

Lisa: On the people side, we're still two engineers short. That $60,000 underspend is essentially
the two open headcount slots we haven't filled. I recommend we carry those requisitions into Q4 and
adjust the forecast upward since the market has tightened.

Marcus: Agreed. Decision: carry both open reqs into Q4, update the plan to $960,000 for Q4 to
account for potential mid-quarter hires. Lisa, can you update the headcount tracker?

Lisa: Will do by Friday.

Marcus: On tooling — we need to renew the observability platform contract. Current contract expires
October 15th. Raj, can you get quotes from two alternatives before renewal date?

Raj: Yes, I'll have comparison ready by October 1st.

Marcus: Good. Final item — the infrastructure cost report. Dan should present it at the all-hands
next Thursday. Dan, are you good with that?

Dan: Yes, I'll prepare a 10-minute deck.

Marcus: Perfect. Let's close there. Summary: Q3 under budget by $60k, two headcount carry to Q4,
observability renewal comparison by October 1st, all-hands infra deck next Thursday.
"""

_G3_TRANSCRIPT = """
Priya: Thanks for making time, Sam. How's the transition to the new role going?

Sam: It's been a lot. I feel like I'm finally getting the rhythm of the product reviews but I'm
still struggling with stakeholder communication. Specifically, engineering leads push back a lot
when I present timelines.

Priya: I've noticed that too. I think part of it is how the timelines are framed — it helps when
estimates come with explicit uncertainty ranges. Have you tried that?

Sam: Not yet. I'll try it in the next roadmap review.

Priya: Also, I'd like to see you take ownership of the weekly engineering sync. Starting next week
you should run that meeting end to end. I'll be there to support but it's your meeting.

Sam: Got it. I'll prep an agenda template.

Priya: One more thing — your performance review is scheduled for November 10th. Before that I'd
like you to submit a self-assessment by October 28th. HR sent the template last week.

Sam: I'll get that done by October 28th.

Priya: Great. Last item — there's a leadership workshop on November 3rd that I think you'd benefit
from. I'll send you the registration link. Sign up by end of this week to get the early slot.

Sam: Perfect, I'll watch for that email.

Priya: Good session. Let's check in again in two weeks.
"""

# ---------------------------------------------------------------------------
# Education — lecture / classroom domain
# ---------------------------------------------------------------------------

_E1_TRANSCRIPT = """
Professor Chen: Today we're covering sorting algorithms — specifically merge sort and quicksort —
and why the choice between them matters in practice.

Merge sort always runs in O(n log n) time, in both best and worst cases. It achieves this by
dividing the array into halves recursively, sorting each half, then merging the sorted halves.
The downside is space: merge sort requires O(n) auxiliary space because you need a temporary
array during the merge step.

Quicksort, by contrast, is an in-place algorithm — it needs only O(log n) stack space for
recursion. Its average time complexity is also O(n log n), but its worst case is O(n squared)
if the pivot selection is poor, for example always picking the first element on an already-sorted
array. Modern implementations use randomized pivot selection or the median-of-three strategy to
avoid this.

In practice, quicksort is often faster than merge sort due to better cache locality — it accesses
memory sequentially during partitioning, which is friendly to modern CPUs.

Student: When would you actually prefer merge sort over quicksort?

Professor Chen: Excellent question. Merge sort is preferred when stability matters — it's a
stable sort, meaning equal elements maintain their original order. Quicksort is not stable by
default. Also, for linked lists, merge sort is generally better because random access is expensive.
For external sorting — when data doesn't fit in memory — merge sort maps naturally to how you'd
merge sorted disk pages.

For your assignment this week, implement both algorithms in Python and compare their runtime on
three dataset shapes: random, sorted, and reverse-sorted. Due Friday.
"""

_E2_TRANSCRIPT = """
Dr. Okafor: Before we dive in, let's recap last session: we covered cell structure and the role
of the nucleus. Today we're going into DNA replication.

DNA replication is the process by which a cell copies its DNA before division. The key enzyme is
DNA polymerase, which reads an existing DNA strand and synthesizes a new complementary strand.
Replication starts at specific locations on the chromosome called origins of replication.

The double helix is unwound by an enzyme called helicase. This creates a replication fork — a
Y-shaped structure where two strands are separated. Because DNA polymerase can only synthesize
in the 5-prime to 3-prime direction, the two strands are copied differently.

The leading strand is synthesized continuously in the direction of the fork. The lagging strand
is synthesized in short fragments called Okazaki fragments, which are later joined by an enzyme
called DNA ligase.

Another important enzyme is primase, which lays down a short RNA primer so DNA polymerase has
a place to start. Those primers are later removed and replaced with DNA.

Student: What happens if there's a mistake during replication?

Dr. Okafor: DNA polymerase has a built-in proofreading activity — it can detect and correct
most errors immediately. The overall error rate is about one mistake per billion base pairs.
Additional mismatch repair systems catch errors that get through.

For the quiz next Tuesday: know the enzymes — helicase, primase, DNA polymerase, DNA ligase —
their roles, and the difference between leading and lagging strand synthesis.
"""

_E3_TRANSCRIPT = """
Ms. Rivera: Good afternoon. Today's lecture covers the causes and key outcomes of the First
Industrial Revolution, roughly 1760 to 1840.

The Industrial Revolution began in Britain for several interconnected reasons. Britain had
abundant coal deposits, especially in Wales and Yorkshire, and iron ore nearby. The enclosure
movement had driven workers off agricultural land into cities, creating a large urban workforce.
Britain also had a stable banking system and could finance new machinery.

The invention of the steam engine by James Watt in 1769 was transformative. It allowed factories
to be built anywhere, not just near rivers. By 1800, steam-powered factories dominated textile
production in Manchester and Birmingham.

The social consequences were significant. Child labor was widespread in textile mills and coal
mines. The average urban worker lived in overcrowded tenements with poor sanitation. Life
expectancy in industrial cities was lower than in rural areas.

However, the period also laid groundwork for the middle class. Factory owners accumulated
capital and political influence. The Reform Act of 1832 extended voting rights partly in
response to the rising industrial bourgeoisie.

Student: Was there any pushback against industrialization?

Ms. Rivera: Yes — the Luddite movement, from roughly 1811 to 1816, saw groups of textile
workers destroy machinery that threatened their livelihoods. They were ultimately suppressed by
the government using military force.

Reading assignment: chapters 4 and 5 of Thompson's The Making of the English Working Class,
due by next Monday. Discussion questions will be on the course portal.
"""

# ---------------------------------------------------------------------------
# Healthcare — clinical / therapy sessions
# ---------------------------------------------------------------------------

_H1_TRANSCRIPT = """
Dr. Patel: Good morning, Mr. Johnson. I'm reviewing your last three months of blood glucose
logs. Your fasting glucose has averaged 148 mg/dL, which is higher than our target of 130.
Your A1C came back at 7.8%, up from 7.2% at your last visit.

Mr. Johnson: I know. The holidays were rough. I wasn't as careful with my diet.

Dr. Patel: I understand. Let's talk about adjustments. I'd like to increase your metformin from
1000 to 1500 mg twice daily. Take it with food to reduce GI side effects. I also want you to
add a 20-minute walk after dinner — even light activity improves post-meal glucose significantly.

Mr. Johnson: I can try the walk. Any dietary changes?

Dr. Patel: Reduce refined carbohydrates — white bread, white rice — and aim for more fiber.
I'll give you a referral to our dietitian, Maria Santos. Please schedule with her within the
next two weeks.

Mr. Johnson: Will do.

Dr. Patel: I'm also ordering a kidney function panel and lipid panel since we haven't done
those in six months. You can do those at the lab on your way out today.

Mr. Johnson: Okay.

Dr. Patel: Let's schedule a follow-up in three months. If your glucose hasn't improved by then,
we may need to consider adding a second medication. Any questions?

Mr. Johnson: No, I think I understand the plan.

Dr. Patel: Good. Take care, and call the office if you're seeing glucose consistently above 200.
"""

_H2_TRANSCRIPT = """
PT Sarah: Hi David. Let's check in on your right knee. How's the pain been this week on a scale
of one to ten?

David: About a four on good days, six on stairs.

PT Sarah: That's improvement from the seven you reported two weeks ago. Let's look at your range
of motion. Can you bend and straighten for me? Good — you're at 95 degrees flexion. Goal is 130.

Today I want to introduce two new exercises to build up the VMO — that's the inner quad muscle
that supports the kneecap. First: terminal knee extensions with the resistance band. Straighten
fully against the band, hold two seconds, slow return. Three sets of fifteen. Second: step-ups
on a four-inch box. One set of ten each leg.

David: Should I do these at home too?

PT Sarah: Yes — the terminal extensions every day, step-ups every other day to allow recovery.
Continue the straight-leg raises and clamshells we added last week.

The ice and elevation after activity is non-negotiable if you want to keep inflammation down.
Twenty minutes, twice a day on your worst days.

David: Got it.

PT Sarah: Your next appointment is Thursday at 10. By then I want you to report your stair
pain specifically. If you can do stairs at a four or below, we'll progress to single-leg work.
Clearance to return to jogging is typically at eight weeks post-injury — you're at five weeks,
so we're on track if you stick to the program.

David: That's encouraging. Thanks Sarah.
"""

_H3_TRANSCRIPT = """
Dr. Williams: Let's discuss Mrs. Chen in room 412. She's a 68-year-old woman admitted three
days ago with a COPD exacerbation. Her O2 saturation has improved to 94% on 2 liters nasal
cannula, up from 88% on admission.

Nurse Kim: She completed her IV methylprednisolone course yesterday. We transitioned her to
oral prednisone 40 mg this morning.

Dr. Williams: Good. Respiratory therapy, how is she doing with the inhaler technique?

RT Maya: She needs coaching — she's actuating before inhaling. We've been working on it.
One more session today should get her there.

Dr. Williams: Important for discharge. Social work, what's the home situation?

SW Alex: She lives alone. Her daughter can come Thursday but not before. There's no home
oxygen currently set up.

Dr. Williams: Then discharge is Thursday at the earliest, contingent on home oxygen being
arranged before she leaves. Alex, please get the home O2 order today and coordinate delivery
for Thursday morning.

SW Alex: I'll place the order this afternoon.

Dr. Williams: Nursing, add a fall risk protocol — her mobility has been limited and prednisone
affects balance in elderly patients.

Nurse Kim: I'll update the care plan now.

Dr. Williams: Plan for discharge Thursday if O2 is confirmed and she's stable at room air
tomorrow. Follow-up with pulmonology in two weeks. Pharmacy, review her home medications for
any interactions with the new oral steroid.

Pharmacist Raj: I'll have a review ready by end of today.
"""

# ---------------------------------------------------------------------------
# Interview — job candidates
# ---------------------------------------------------------------------------

_I1_TRANSCRIPT = """
Interviewer: Thanks for coming in, Jordan. Let's start with a system design question. Design
a URL shortener like bit.ly. Walk me through your approach.

Jordan: Sure. At the core I need to generate short codes and map them to original URLs. I'd
start with a simple hashing scheme — take the URL, hash it, take the first 6 characters. For
storage, a key-value store like Redis would work well for low-latency reads, with a relational
database for persistence and analytics.

Interviewer: How would you handle collisions in your hash?

Jordan: Good point. I'd use a consistent hash function like MurmurHash, and if a collision
occurs — same 6-character code for a different URL — I'd try the next 6 characters of the hash.
Alternatively, I'd maintain a counter in the database and encode it in base62 to generate the
short code, which guarantees uniqueness without collisions.

Interviewer: That's better. What about scale — say 1 billion URLs?

Jordan: At that scale I'd shard the database by the first character of the short code. I'd
also add a CDN layer to serve redirects for hot URLs without hitting the backend at all. For
write throughput I'd consider an async write path — put the mapping in Redis immediately and
write to the database in a background job.

Interviewer: Any concerns with that async approach?

Jordan: Yes — if the service crashes before the background write, we lose the mapping. I'd
mitigate that with a write-ahead log or by using Redis' persistence features. In practice,
bit.ly probably accepts a very small window of potential URL loss to get the write throughput.

Interviewer: Great. One behavioral question — tell me about a time you disagreed with a
technical decision on your team.

Jordan: At my last company we were considering moving from PostgreSQL to MongoDB for our
main product database. I disagreed — our data was relational with complex joins and MongoDB
would have hurt us. I prepared a comparison document and presented it to the team. We ran
a proof of concept and ultimately kept PostgreSQL with some schema optimizations instead.
"""

_I2_TRANSCRIPT = """
Interviewer: Hi Maya. I'd like to start with a product case. You're a PM at a food delivery
company. Engagement is down 15% month over month. How do you investigate and address this?

Maya: First I'd want to understand whether this is a metric definition issue or a real signal.
I'd check: is engagement measured by orders, app opens, or something else? Then I'd segment
the drop by user cohort — new users vs retained, geography, platform. Often a drop in one
segment explains the overall number.

Assuming it's real, I'd look at the funnel: where are users dropping off? If it's at the
restaurant selection stage, the issue could be search relevance or selection quality. If it's
at checkout, it could be price or delivery time expectation.

I'd also check external factors — did a competitor launch something? Did we change pricing?

Interviewer: Say it's a new cohort retention problem — users acquired in the last 60 days
aren't coming back after their first order. What do you do?

Maya: That points to a first-order experience issue. I'd look at ratings and reviews from
that cohort specifically. If they had late deliveries or wrong items, that kills retention.
I'd also look at the push notification strategy — are we re-engaging them appropriately?

I'd propose a targeted re-engagement campaign: a discount on their second order, sent 3 to
5 days after their first. Paired with improvements to delivery time accuracy so the
expectation is set correctly at checkout.

Interviewer: How would you measure success?

Maya: Primary metric: 30-day retention for new cohorts improves by 5 percentage points. Guard
rail metrics: margin per order stays flat (we're not just buying retention with discounts),
and customer support ticket volume doesn't increase.
"""

_I3_TRANSCRIPT = """
Interviewer: Welcome, Priya. Let's start with a technical question. You have a dataset of
10 million customer transactions. You need to detect anomalous transactions — possible fraud.
How do you approach this?

Priya: I'd frame it as an anomaly detection problem, not a classification problem — because
fraud labels are rare and often noisy. My first step is exploratory: understand the distribution
of transaction amounts, frequencies, merchant categories, and time patterns.

For a baseline, I'd start with statistical methods — flag transactions more than three standard
deviations from a user's historical mean, or from the distribution of similar users in the
same merchant category.

For a more robust approach, I'd train an isolation forest or autoencoder on the normal
transaction data — these models learn what normal looks like and flag deviations. The advantage
over supervised models is I don't need accurate fraud labels.

Interviewer: What if you have some labels — 5,000 confirmed fraud cases out of 10 million?

Priya: Then I'd use a semi-supervised approach. Train a supervised classifier on the labeled
data — gradient boosting works well here — but also incorporate the unsupervised anomaly score
as a feature. The ensemble often outperforms either alone.

With imbalanced classes at 0.05%, I'd oversample the minority class with SMOTE or tune the
class weight parameter. I'd evaluate on precision-recall AUC rather than accuracy.

Interviewer: How would you deploy this in production?

Priya: Real-time scoring for each transaction via a lightweight model served by a REST API —
decision tree or logistic regression for sub-millisecond latency. The heavier models run in
batch overnight to retrain and recalibrate. I'd also implement a feedback loop: when fraud is
confirmed by the investigation team, those labels go back into the training set automatically.
"""

# ---------------------------------------------------------------------------
# Project — sprint retrospectives, decisions, postmortems
# ---------------------------------------------------------------------------

_P1_TRANSCRIPT = """
Scrum Master Tom: Let's start the retro. Sprint 24 was eventful. What went well?

Dev Anna: The new CI pipeline we set up in Sprint 23 really paid off — no failed deploys this
sprint. We caught two regression bugs in staging before they hit prod.

Dev Kwame: The pair programming sessions helped. I got unstuck on the auth service refactor much
faster with a second set of eyes.

Tom: Good. What could have gone better?

Dev Anna: Story estimation is still off. We took on 42 points and completed 31. The OAuth
integration turned out to be twice as complex as estimated.

QA Mia: We had three stories arrive in QA with incomplete acceptance criteria. I spent half a
day going back to product to clarify. That needs to stop.

Product Owner Lena: My fault on the criteria gaps. I'll add a definition-of-ready checklist to
the story template. Stories without clear acceptance criteria won't be allowed into the sprint.

Tom: Decision: definition-of-ready checklist enforced from Sprint 25 onwards. Lena owns the
template by next Monday.

Dev Kwame: Also, we need a postmortem for the API gateway timeout incident on day 3. It affected
three customers. Anna and I should own that.

Tom: Agreed. Schedule the postmortem for Wednesday. Action item: Kwame and Anna run postmortem
and share findings with the team by Friday.

Tom: Velocity: three-sprint average is now 33 points. I recommend we cap Sprint 25 at 34 points.
All agree?

All: Agreed.

Tom: Good sprint. Let's close there.
"""

_P2_TRANSCRIPT = """
Tech Lead Sam: We need to decide on the database technology for the new analytics service.
The team has evaluated three options: PostgreSQL with partitioning, ClickHouse, and BigQuery.

Eng Diana: I ran benchmarks on our expected query patterns — complex aggregations over 12 months
of event data. ClickHouse was significantly faster: 2.3 seconds for the worst-case query vs
14 seconds on partitioned PostgreSQL. BigQuery was 1.8 seconds but has per-query cost at scale.

Eng Ravi: Operational complexity matters too. ClickHouse requires us to manage the cluster.
BigQuery is fully managed. PostgreSQL we already know.

Sam: What's our expected query volume?

Diana: About 500 dashboard queries per day from internal users. Not massive.

Ravi: At that volume, BigQuery cost would be roughly $80 to $120 per month. Manageable.

Sam: So the options are: ClickHouse for performance at operational cost, or BigQuery for managed
simplicity at monetary cost.

Diana: I lean ClickHouse — we'll hit scale where query performance matters and our data doesn't
leave our own infrastructure.

Ravi: I'd go BigQuery for the first six months. We can migrate if we outgrow it.

Sam: Decision: we go with ClickHouse. Rationale — we have the operational maturity and the
performance characteristics fit our long-term roadmap. We don't want to migrate later when data
volumes are higher.

Action items: Diana sets up the ClickHouse cluster in staging by end of sprint. Ravi documents
the schema design and data pipeline by next Friday. Sam writes up the ADR and posts to Confluence.
"""

_P3_TRANSCRIPT = """
Incident Commander Riya: This postmortem covers the payment service outage on December 2nd,
from 14:07 to 15:43 UTC — 96 minutes of impact. About 2,400 transactions failed.

SRE Ben: Timeline: at 14:05 we deployed payment service v2.4.1. At 14:07 error rate spiked
to 40%. At 14:15 we were paged. At 14:30 we identified the root cause — a missing database
index on the payments table for the new order lookup query. At 14:50 we rolled back to v2.4.0.
Full recovery at 15:43 after connection pools drained.

Dev Mia: Root cause: the new query in v2.4.1 does a full table scan on payments without an
index. In staging it was fine because the staging dataset is 10,000 rows. Prod has 45 million
rows. The full scan caused connection pool exhaustion within 2 minutes.

Riya: What should have caught this?

Mia: We should have had query explain plan review in our release checklist. We didn't. Also,
our staging database should have a representative data volume — 10,000 rows is not representative.

Riya: Action items. First: add explain plan review to the deployment checklist. Mia owns this
by December 9th. Second: scale the staging database to at least 1 million rows. Ben owns
the script by December 16th. Third: add a query performance alert that fires when p95 query
time exceeds 500ms. SRE team owns this by December 12th.

Ben: Also recommend we implement a circuit breaker on the payment service so future degradation
doesn't cascade to full failure. That's a longer item — target end of Q1.

Riya: Agreed, add to Q1 backlog. Decision: staging data volume is now a P1 infrastructure gap.
Any objections to the action items? None — meeting closed.
"""

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_CASES: list[EvalCase] = [
    EvalCase(
        id="general-standup",
        domain="General",
        title="Team Weekly Standup",
        source="synthetic",
        transcript=_G1_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "Alice finished the user dashboard component",
                "Bob deployed payment service to staging",
                "Carol filed ticket MN-412",
                "Carol is blocked on iOS 17 token expiry",
                "sprint demo is Friday at 2pm",
            ],
            "action_owners": ["Alice", "Carol", "Dan"],
            "action_tasks": ["ping Marcus from iOS team", "send updated endpoint spec", "features in staging by Thursday"],
        },
    ),
    EvalCase(
        id="general-budget",
        domain="General",
        title="Q3 Budget Review",
        source="synthetic",
        transcript=_G2_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "Q3 budget was 900,000",
                "Q3 actuals were 840,000",
                "cloud spend reduction of 22,000",
                "two open headcount slots",
                "observability contract expires October 15th",
                "all-hands infrastructure deck next Thursday",
            ],
            "action_owners": ["Lisa", "Raj", "Dan"],
            "action_tasks": ["update headcount tracker", "get quotes for observability alternatives", "prepare infrastructure deck"],
        },
    ),
    EvalCase(
        id="general-oneonone",
        domain="General",
        title="Manager 1:1 — Onboarding Check-in",
        source="synthetic",
        transcript=_G3_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "Sam struggling with stakeholder communication",
                "performance review November 10th",
                "self-assessment due October 28th",
                "leadership workshop November 3rd",
                "Sam to run weekly engineering sync starting next week",
            ],
            "action_owners": ["Sam", "Priya"],
            "action_tasks": ["submit self-assessment", "register for leadership workshop", "prepare agenda template for engineering sync"],
        },
    ),
    EvalCase(
        id="education-sorting",
        domain="Education",
        title="CS Lecture: Sorting Algorithms",
        source="synthetic",
        transcript=_E1_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "merge sort runs in O(n log n) in all cases",
                "merge sort requires O(n) auxiliary space",
                "quicksort worst case is O(n squared)",
                "quicksort is in-place with O(log n) stack space",
                "quicksort has better cache locality",
                "merge sort is a stable sort",
                "assignment due Friday",
            ],
            "key_concepts": ["merge sort", "quicksort", "O(n log n)", "stable sort", "cache locality"],
            "learning_objectives": ["compare time and space complexity of merge sort and quicksort"],
        },
    ),
    EvalCase(
        id="education-dna",
        domain="Education",
        title="Biology Lecture: DNA Replication",
        source="synthetic",
        transcript=_E2_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "DNA polymerase synthesizes new complementary strand",
                "helicase unwinds the double helix",
                "leading strand synthesized continuously",
                "lagging strand uses Okazaki fragments",
                "DNA ligase joins Okazaki fragments",
                "primase lays down RNA primer",
                "error rate is one per billion base pairs",
                "quiz on Tuesday",
            ],
            "key_concepts": ["DNA polymerase", "helicase", "leading strand", "lagging strand", "Okazaki fragments", "DNA ligase", "primase"],
        },
    ),
    EvalCase(
        id="education-industrial",
        domain="Education",
        title="History Lecture: Industrial Revolution",
        source="synthetic",
        transcript=_E3_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "Industrial Revolution began in Britain 1760 to 1840",
                "James Watt invented steam engine in 1769",
                "Luddite movement from 1811 to 1816",
                "Reform Act of 1832 extended voting rights",
                "enclosure movement drove workers into cities",
                "reading assignment due next Monday",
            ],
            "key_concepts": ["Industrial Revolution", "steam engine", "Luddite movement", "Reform Act of 1832"],
        },
    ),
    EvalCase(
        id="healthcare-diabetes",
        domain="Healthcare",
        title="Diabetes Follow-up Appointment",
        source="synthetic",
        transcript=_H1_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "fasting glucose averaged 148 mg/dL",
                "target fasting glucose 130 mg/dL",
                "A1C is 7.8%",
                "metformin increased to 1500 mg twice daily",
                "referral to dietitian Maria Santos",
                "kidney function panel ordered",
                "lipid panel ordered",
                "follow-up in three months",
            ],
            "action_owners": ["Mr. Johnson"],
            "action_tasks": ["schedule with dietitian within two weeks", "do lab work today", "20-minute walk after dinner"],
        },
    ),
    EvalCase(
        id="healthcare-physicaltherapy",
        domain="Healthcare",
        title="Physical Therapy: Knee Rehabilitation",
        source="synthetic",
        transcript=_H2_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "pain level four to six this week",
                "range of motion at 95 degrees flexion",
                "goal is 130 degrees flexion",
                "next appointment Thursday at 10",
                "return to jogging target at eight weeks",
                "patient is at five weeks post-injury",
            ],
            "action_owners": ["David"],
            "action_tasks": ["terminal knee extensions every day", "step-ups every other day", "ice and elevation 20 minutes twice daily"],
        },
    ),
    EvalCase(
        id="healthcare-careteam",
        domain="Healthcare",
        title="Care Team Rounds — COPD Exacerbation",
        source="synthetic",
        transcript=_H3_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "Mrs. Chen is 68 years old",
                "O2 saturation improved to 94% on 2 liters",
                "transitioned to oral prednisone 40 mg",
                "discharge planned for Thursday",
                "home oxygen needs to be arranged",
                "pulmonology follow-up in two weeks",
                "fall risk protocol added",
            ],
            "action_owners": ["Alex", "Kim", "Raj"],
            "action_tasks": ["place home oxygen order today", "update care plan for fall risk", "medication interaction review by end of day"],
        },
    ),
    EvalCase(
        id="interview-swe",
        domain="Interview",
        title="Software Engineer — System Design Interview",
        source="synthetic",
        transcript=_I1_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "Jordan proposed key-value store for URL shortener",
                "Jordan mentioned MurmurHash",
                "Jordan suggested base62 encoding with counter",
                "Jordan proposed CDN layer for hot URLs",
                "Jordan mentioned write-ahead log for async writes",
                "Jordan disagreed with MongoDB migration at previous company",
            ],
            "red_flags": [],
            "green_flags": ["handles collisions correctly", "addresses scale proactively", "acknowledges trade-offs"],
        },
    ),
    EvalCase(
        id="interview-pm",
        domain="Interview",
        title="Product Manager — Product Sense Interview",
        source="synthetic",
        transcript=_I2_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "engagement down 15% month over month",
                "Maya proposed segmenting by user cohort",
                "Maya identified new cohort retention problem",
                "proposed re-engagement campaign with discount",
                "primary metric 30-day retention improvement by 5 percentage points",
            ],
            "green_flags": ["structured problem decomposition", "proposed guard rail metrics"],
        },
    ),
    EvalCase(
        id="interview-ds",
        domain="Interview",
        title="Data Scientist — Technical and Case Interview",
        source="synthetic",
        transcript=_I3_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "dataset of 10 million customer transactions",
                "Priya suggested isolation forest and autoencoder",
                "5000 confirmed fraud cases out of 10 million",
                "semi-supervised approach with gradient boosting",
                "precision-recall AUC preferred over accuracy",
                "lightweight model for sub-millisecond latency",
                "feedback loop for confirmed fraud labels",
            ],
            "green_flags": ["correctly identifies class imbalance challenge", "proposes feedback loop"],
        },
    ),
    EvalCase(
        id="project-retro",
        domain="Project",
        title="Sprint 24 Retrospective",
        source="synthetic",
        transcript=_P1_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "Sprint 24 completed 31 of 42 story points",
                "three stories arrived in QA with incomplete acceptance criteria",
                "definition-of-ready checklist decision",
                "three-sprint average velocity is 33 points",
                "Sprint 25 capped at 34 points",
                "postmortem scheduled for Wednesday",
            ],
            "action_owners": ["Lena", "Kwame", "Anna"],
            "action_tasks": ["definition-of-ready template by next Monday", "postmortem findings shared by Friday"],
            "decisions": ["definition-of-ready checklist from Sprint 25", "Sprint 25 cap at 34 points"],
        },
    ),
    EvalCase(
        id="project-architecture",
        domain="Project",
        title="Analytics Database Architecture Decision",
        source="synthetic",
        transcript=_P2_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "three options evaluated: PostgreSQL, ClickHouse, BigQuery",
                "ClickHouse 2.3 seconds worst-case query",
                "PostgreSQL 14 seconds worst-case query",
                "BigQuery 1.8 seconds worst-case query",
                "BigQuery cost 80 to 120 per month",
                "decision: ClickHouse selected",
                "500 dashboard queries per day",
            ],
            "action_owners": ["Diana", "Ravi", "Sam"],
            "action_tasks": ["ClickHouse cluster in staging by end of sprint", "schema design by next Friday", "write ADR and post to Confluence"],
            "decisions": ["ClickHouse selected for analytics service"],
        },
    ),
    EvalCase(
        id="project-postmortem",
        domain="Project",
        title="Payment Service Outage Postmortem",
        source="synthetic",
        transcript=_P3_TRANSCRIPT.strip(),
        ground_truth={
            "facts": [
                "outage on December 2nd from 14:07 to 15:43 UTC",
                "96 minutes of impact",
                "2400 transactions failed",
                "root cause was missing database index",
                "staging database had only 10,000 rows",
                "production database has 45 million rows",
                "rolled back to v2.4.0",
            ],
            "action_owners": ["Mia", "Ben", "SRE team"],
            "action_tasks": ["add explain plan to deployment checklist", "scale staging database to 1 million rows", "add query performance alert"],
            "decisions": ["staging data volume is P1 infrastructure gap", "circuit breaker added to Q1 backlog"],
        },
    ),
]


def get_cases(domain: str | None = None) -> list[EvalCase]:
    if domain:
        return [c for c in ALL_CASES if c.domain == domain]
    return ALL_CASES


def get_case(case_id: str) -> EvalCase | None:
    for c in ALL_CASES:
        if c.id == case_id:
            return c
    return None
