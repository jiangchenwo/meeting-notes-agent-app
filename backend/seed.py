from database import SessionLocal, engine
from models import Base, Domain, NoteBlock, Template


DOMAINS = [
    ("Education", "Lecture notes, study guides, concepts, assignments, and quizzes"),
    ("Healthcare", "Healthcare and therapy draft notes, care tasks, and review flags"),
    ("Interview", "Mock and real interview notes, coaching, evidence, and follow-ups"),
    ("Project", "Project meeting decisions, action items, blockers, risks, and dependencies"),
    ("General", "Flexible notes, action items, and key points for any type of conversation"),
]

TEMPLATES = [
    ("Lecture Notes", "Education", "Lecture notes with key concepts, objectives, assignments, and quiz questions.", "Extract lecture notes, key concepts, learning objectives, assignments, and quiz questions."),
    ("Healthcare Notes", "Healthcare", "Session notes with concerns, observations, care tasks, risk flags, and review notes.", "Summarize the session, document key concerns and observations, list care tasks and follow-up actions, flag any risks, and add review notes."),
    ("Interview Notes", "Interview", "Interview notes with questions, strengths, concerns, follow-ups, and coaching.", "Extract questions asked, evidence-backed strengths, concerns, follow-ups, and coaching notes."),
    ("Project Meeting", "Project", "Project notes with decisions, action items, blockers, risks, and dependencies.", "Extract decisions, action items, blockers, risks, dependencies, owners, and deadlines."),
    ("General Meeting Notes", "General", "General notes with a summary, key points, decisions, and action items.", "Summarize the conversation, extract action items and decisions, and list the key points discussed."),
]


def sync_builtin_domains_and_templates(db) -> dict[str, int]:
    changed = {"domains": 0, "templates": 0, "removed_domains": 0, "removed_templates": 0}

    for sort_order, (name, description) in enumerate(DOMAINS, start=1):
        domain = db.query(Domain).filter(Domain.name == name).first()
        if domain:
            domain.description = description
            domain.is_builtin = True
            domain.sort_order = sort_order
        else:
            db.add(Domain(name=name, description=description, is_builtin=True, sort_order=sort_order))
            changed["domains"] += 1
    db.flush()

    domain_map = {d.name: d for d in db.query(Domain).all()}
    for name, domain_name, description, prompt in TEMPLATES:
        template = db.query(Template).filter(Template.name == name).first()
        if template:
            template.domain_id = domain_map[domain_name].id
            template.description = description
            template.prompt_template = prompt
            template.is_builtin = True
        else:
            db.add(Template(
                name=name,
                domain_id=domain_map[domain_name].id,
                description=description,
                prompt_template=prompt,
                is_builtin=True,
            ))
            changed["templates"] += 1
    db.flush()

    template_names = {name for name, _, _, _ in TEMPLATES}
    obsolete_templates = (
        db.query(Template)
        .filter(Template.is_builtin.is_(True))
        .filter(~Template.name.in_(template_names))
        .all()
    )
    for template in obsolete_templates:
        db.query(NoteBlock).filter(NoteBlock.template_id == template.id).update({"template_id": None})
        db.delete(template)
        changed["removed_templates"] += 1

    domain_names = {name for name, _ in DOMAINS}
    project_domain = domain_map["Project"]
    obsolete_domains = (
        db.query(Domain)
        .filter(Domain.is_builtin.is_(True))
        .filter(~Domain.name.in_(domain_names))
        .all()
    )
    for domain in obsolete_domains:
        db.query(NoteBlock).filter(NoteBlock.domain_id == domain.id).update({"domain_id": project_domain.id})
        db.query(Template).filter(Template.domain_id == domain.id).update({"domain_id": None})
        db.delete(domain)
        changed["removed_domains"] += 1

    return changed


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        sync_builtin_domains_and_templates(db)
        db.commit()
        print("Database synced with priority domains and templates.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
