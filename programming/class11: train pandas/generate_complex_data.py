"""Воспроизводимо создаёт CSV для комплексных задач 26–30."""

from datetime import datetime, timedelta
from pathlib import Path
import random

import pandas as pd


OUT = Path(__file__).parent / "complex_data"
OUT.mkdir(exist_ok=True)
rng = random.Random(20261126)


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)


# 26. Устройства: одинаковые моменты, посторонние события и устройства без ошибок.
events = []
event_id = 1
for device in [f"D{i:02d}" for i in range(1, 9)]:
    for _ in range(rng.randint(7, 15)):
        minute = rng.randrange(0, 181)
        kind = rng.choices(["error", "reset", "heartbeat"], [4, 3, 5])[0]
        events.append(dict(event_id=event_id, device_id=device,
                           timestamp=datetime(2026, 10, 15, 9) + timedelta(minutes=minute),
                           event=kind))
        event_id += 1
events += [
    dict(event_id=event_id, device_id="D01", timestamp="2026-10-15 10:00:00", event="error"),
    dict(event_id=event_id + 1, device_id="D01", timestamp="2026-10-15 10:00:00", event="reset"),
    dict(event_id=event_id + 2, device_id="D09", timestamp="2026-10-15 09:30:00", event="heartbeat"),
]
requests = []
for i in range(1, 151):
    requests.append(dict(request_id=i, device_id=f"D{rng.randrange(1, 11):02d}",
                         timestamp=datetime(2026, 10, 15, 9)
                         + timedelta(minutes=rng.randrange(0, 181))))
requests += [dict(request_id=151, device_id="D01", timestamp="2026-10-15 10:00:00"),
             dict(request_id=152, device_id="D10", timestamp="2026-10-15 09:00:00")]
rng.shuffle(events)
rng.shuffle(requests)
save("device_events.csv", events)
save("device_requests.csv", requests)


# 27. Заказы: несколько позиций и частичные возвраты до и после даты отчёта.
orders, items, returns = [], [], []
item_id = return_id = 1
for order_id in range(1, 91):
    status = "cancelled" if order_id % 11 == 0 else "paid"
    created = datetime(2026, 10, 1) + timedelta(days=rng.randrange(30))
    orders.append(dict(order_id=order_id, customer_id=1000 + rng.randrange(1, 38),
                       created_at=created, status=status))
    for _ in range(rng.randint(1, 5)):
        quantity = rng.randint(1, 8)
        items.append(dict(item_id=item_id, order_id=order_id,
                          product_id=f"P{rng.randrange(1, 31):03d}", quantity=quantity,
                          unit_price=rng.randrange(4, 101) * 10,
                          discount_pct=rng.choice([0, 0, 10, 20])))
        if status == "paid" and item_id % 3 == 0:
            returned = rng.randint(1, quantity)
            first = rng.randint(1, returned)
            first_at = created + timedelta(days=rng.randrange(1, 20))
            returns.append(dict(return_id=return_id, item_id=item_id, quantity=first,
                                returned_at=first_at))
            return_id += 1
            if returned > first:
                returns.append(dict(return_id=return_id, item_id=item_id,
                                    quantity=returned - first,
                                    returned_at=first_at + timedelta(days=rng.randrange(0, 11))))
                return_id += 1
        item_id += 1
rng.shuffle(items)
rng.shuffle(returns)
save("orders.csv", orders)
save("order_items.csv", items)
save("item_returns.csv", returns)


# 28. Библиотека: продления, возвраты и события после отчётной даты.
loans, loan_events = [], []
action_id = 1
for loan_id in range(1, 81):
    borrowed = datetime(2026, 9, 1) + timedelta(days=rng.randrange(48))
    due = borrowed + timedelta(days=rng.choice([14, 21, 28]))
    loans.append(dict(loan_id=loan_id, reader_id=2000 + rng.randrange(1, 29),
                      book_id=f"B{rng.randrange(1, 41):03d}",
                      borrowed_at=borrowed, due_at=due.date()))
    if loan_id % 3 != 0:
        at = borrowed + timedelta(days=rng.randrange(4, 18), hours=10)
        loan_events.append(dict(action_id=action_id, loan_id=loan_id, timestamp=at,
                                action="renew", new_due_at=(due + timedelta(days=14)).date()))
        action_id += 1
    if loan_id % 4 != 0:
        at = borrowed + timedelta(days=rng.randrange(12, 42), hours=15)
        loan_events.append(dict(action_id=action_id, loan_id=loan_id, timestamp=at,
                                action="return", new_due_at=""))
        action_id += 1
    if loan_id % 9 == 0:
        loan_events.append(dict(action_id=action_id, loan_id=loan_id,
                                timestamp="2026-11-05 12:00:00", action="renew",
                                new_due_at="2026-12-01"))
        action_id += 1
rng.shuffle(loan_events)
save("loans.csv", loans)
save("loan_events.csv", loan_events)


# 29. Склад: остатки, движения и запросы о наличии.
opening, movements, stock_requests = [], [], []
movement_id = 1
for warehouse in ["W1", "W2", "W3"]:
    for sku in ["A", "B", "C", "D"]:
        opening.append(dict(warehouse=warehouse, sku=sku, quantity=rng.randrange(10, 41)))
        for _ in range(rng.randint(8, 15)):
            movements.append(dict(movement_id=movement_id, warehouse=warehouse, sku=sku,
                                  timestamp=datetime(2026, 10, 20, 8)
                                  + timedelta(minutes=rng.randrange(0, 301)),
                                  delta=rng.randrange(-5, 9)))
            movement_id += 1
for request_id in range(1, 151):
    stock_requests.append(dict(request_id=request_id,
                               warehouse=rng.choice(["W1", "W2", "W3"]),
                               sku=rng.choice(["A", "B", "C", "D"]),
                               timestamp=datetime(2026, 10, 20, 8)
                               + timedelta(minutes=rng.randrange(0, 301)),
                               quantity=rng.randrange(1, 45)))
rng.shuffle(movements)
rng.shuffle(stock_requests)
save("opening_stock.csv", opening)
save("stock_movements.csv", movements)
save("stock_requests.csv", stock_requests)


# 30. Поддержка: ответы, закрытия, повторные открытия и события после среза.
tickets, ticket_events = [], []
ticket_event_id = 1
for ticket_id in range(1, 101):
    opened = datetime(2026, 10, 30, 8) + timedelta(minutes=rng.randrange(0, 3000))
    priority = rng.choice(["high", "normal", "low"])
    tickets.append(dict(ticket_id=ticket_id, priority=priority, opened_at=opened))
    if ticket_id % 8 != 0:
        ticket_events.append(dict(event_id=ticket_event_id, ticket_id=ticket_id,
                                  timestamp=opened + timedelta(minutes=rng.randrange(10, 1500)),
                                  event="agent_reply"))
        ticket_event_id += 1
    if ticket_id % 3 != 0:
        closed = opened + timedelta(minutes=rng.randrange(1500, 3500))
        ticket_events.append(dict(event_id=ticket_event_id, ticket_id=ticket_id,
                                  timestamp=closed, event="close"))
        ticket_event_id += 1
        if ticket_id % 5 == 0:
            ticket_events.append(dict(event_id=ticket_event_id, ticket_id=ticket_id,
                                      timestamp=closed + timedelta(minutes=10), event="reopen"))
            ticket_event_id += 1
    if ticket_id % 7 == 0:
        ticket_events.append(dict(event_id=ticket_event_id, ticket_id=ticket_id,
                                  timestamp=opened + timedelta(minutes=100),
                                  event="customer_message"))
        ticket_event_id += 1
rng.shuffle(ticket_events)
save("tickets.csv", tickets)
save("ticket_events.csv", ticket_events)

print(f"Создано {len(list(OUT.glob('*.csv')))} CSV в {OUT}")
