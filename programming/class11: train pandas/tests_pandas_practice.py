"""Проверка ответов заданий из pandas-practice.ipynb.

В ячейке задания вызовите test_task_N(номер, ans). В ans передаётся готовый
результат, а не функция. Графики, объяснения и выводы проверяются преподавателем.
Для задач 1–25 эталон строится из data/online_retail.csv.gz, который создаёт
ноутбук. Для задач 26–30 используются CSV из папки complex_data рядом с ним.
"""

from functools import lru_cache
from pathlib import Path
import math

import numpy as np
import pandas as pd

__all__ = ["test_task_N"]


def _data_path():
    candidates = [Path.cwd() / "data/online_retail.csv.gz",
                  Path(__file__).parent / "data/online_retail.csv.gz"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("Сначала запустите ячейку загрузки данных в ноутбуке")
    return path


@lru_cache(maxsize=1)
def _data():
    path = _data_path()
    retail = pd.read_csv(
        path, compression="gzip", parse_dates=["InvoiceDate"],
        dtype={"InvoiceNo": "string", "StockCode": "string",
               "Description": "string", "CustomerID": "Int64", "Country": "string"},
        low_memory=False,
    )
    base = retail.drop_duplicates().copy()
    sales = base.loc[
        ~base["InvoiceNo"].str.startswith("C", na=False)
        & base["Quantity"].gt(0) & base["UnitPrice"].gt(0)
    ].copy()
    sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]
    return retail, base, sales


def _value(frame):
    return frame["Quantity"] * frame["UnitPrice"]


def _country_table(sales):
    result = sales.groupby("Country").agg(
        lines=("InvoiceNo", "size"), orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"), revenue=("Revenue", "sum"))
    result["share"] = result["revenue"] / result["revenue"].sum()
    return result.sort_index()


def _month_table(sales):
    month = sales["InvoiceDate"].dt.to_period("M").astype(str)
    result = sales.assign(month=month).groupby("month").agg(
        revenue=("Revenue", "sum"), orders=("InvoiceNo", "nunique"))
    result["per_order"] = result["revenue"] / result["orders"]
    return result.sort_index()


def _daily(sales):
    daily = sales.set_index("InvoiceDate")["Revenue"].resample("D").sum(min_count=1)
    result = daily.to_frame("revenue")
    result["prev_day"] = daily.shift(1)
    result["prev_7d_mean"] = daily.shift(1).rolling(7, min_periods=3).mean()
    result["cumulative_before"] = daily.cumsum().shift(1)
    return result


@lru_cache(maxsize=30)
def _expected(number):
    if number >= 26:
        return _expected_complex(number)
    retail, base, sales = _data()
    cancel = retail["InvoiceNo"].str.startswith("C", na=False)
    positive = ~cancel & retail["Quantity"].gt(0) & retail["UnitPrice"].gt(0)

    if number == 1:
        return {
            "rows": len(retail), "columns": retail.shape[1],
            "duplicates": int(retail.duplicated().sum()),
            "audit": pd.DataFrame({
                "dtype": retail.dtypes.astype(str), "missing": retail.isna().sum(),
                "missing_share": retail.isna().mean(), "unique": retail.nunique(),
            }),
        }
    if number == 2:
        monthly = retail["InvoiceDate"].dt.to_period("M").value_counts().sort_index()
        monthly.index = monthly.index.astype(str)
        return {"first_date": retail["InvoiceDate"].min(),
                "last_date": retail["InvoiceDate"].max(), "monthly_lines": monthly}
    if number == 3:
        sample = pd.read_csv(_data_path(), nrows=10000)
        return {"sample_types": sample.dtypes.astype(str),
                "retail_types": retail.dtypes.astype(str)}
    if number == 4:
        cols = ["Quantity", "UnitPrice"]
        result = retail[cols].quantile([.5, .99, .999]).T
        result.columns = ["median", "p99", "p999"]
        result["omitted_above_p99"] = [int(retail[c].gt(result.loc[c, "p99"]).sum())
                                        for c in cols]
        return result
    if number == 5:
        first = retail.head(20)[["Quantity", "UnitPrice"]]
        joined = pd.concat([retail.loc[0:2, "Quantity"],
                            retail.loc[1:3, "UnitPrice"]], axis=1)
        return {"column_sums": first.sum(axis=0),
                "row_sums": first.sum(axis=1), "joined": joined}
    if number == 6:
        kind = np.select([cancel, positive], ["cancel", "positive_sale"], default="other")
        result = pd.DataFrame({"kind": kind, "value": _value(retail)})
        return result.groupby("kind").agg(lines=("value", "size"),
                                           value=("value", "sum")).sort_index()
    if number == 7:
        countries = ["United Kingdom", "Netherlands", "EIRE"]
        months = ["2011-06", "2011-11"]
        month = retail["InvoiceDate"].dt.to_period("M").astype(str)
        return {
            "known_lines": int(retail["CustomerID"].notna().sum()),
            "unknown_lines": int(retail["CustomerID"].isna().sum()),
            "unknown_share": float(retail["CustomerID"].isna().mean()),
            "country_missing_share": retail.groupby("Country")["CustomerID"].apply(
                lambda x: x.isna().mean()).reindex(countries),
            "month_missing_share": retail.groupby(month)["CustomerID"].apply(
                lambda x: x.isna().mean()).reindex(months),
        }
    if number == 8:
        subset = retail.loc[
            retail["Country"].isin(["United Kingdom", "Netherlands"])
            & retail["UnitPrice"].gt(10),
            ["InvoiceNo", "Country", "UnitPrice"]]
        return {"first_label": int(subset.index[0]),
                "first_five": subset.head(5),
                "reset_first_five": subset.reset_index(drop=True).head(5)}
    if number == 9:
        labels = np.select(
            [cancel, retail["Quantity"].le(0), retail["UnitPrice"].le(0)],
            ["cancel", "nonpositive_quantity", "nonpositive_price"],
            default="positive_sale")
        return pd.Series(labels, index=retail.index, name="kind")
    if number == 10:
        lines = retail.loc[positive, ["InvoiceNo", "StockCode", "Quantity", "UnitPrice"]].copy()
        lines["Revenue"] = lines["Quantity"] * lines["UnitPrice"]
        return lines.sort_values("Revenue", ascending=False, kind="stable").head(10)
    if number == 11:
        stages = [retail, base,
                  base.loc[~base["InvoiceNo"].str.startswith("C", na=False)]]
        stages.append(stages[-1].loc[stages[-1]["Quantity"].gt(0)])
        stages.append(stages[-1].loc[stages[-1]["UnitPrice"].gt(0)])
        return pd.DataFrame({
            "rows": [len(x) for x in stages],
            "value": [float(_value(x).sum()) for x in stages],
        }, index=["raw", "deduplicated", "without_cancel", "positive_quantity",
                  "positive_price"])
    if number == 12:
        dup = retail.duplicated()
        raw_positive = retail.loc[positive]
        month = retail["InvoiceDate"].dt.to_period("M").astype(str)
        return {
            "duplicate_lines": int(dup.sum()),
            "country_duplicates": retail.loc[dup].groupby("Country").size().reindex(
                ["United Kingdom", "Netherlands"], fill_value=0),
            "month_duplicates": retail.loc[dup].groupby(month[dup]).size().reindex(
                ["2011-06", "2011-11"], fill_value=0),
            "revenue_removed": float(_value(raw_positive).sum() - sales["Revenue"].sum()),
        }
    if number == 13:
        names = sorted(retail.loc[retail["StockCode"].eq("22776"),
                                  "Description"].dropna().unique().tolist())
        return {"missing_lines": int(retail["Description"].isna().sum()),
                "cancel_missing_share": float(retail.loc[cancel, "Description"].isna().mean()),
                "sale_missing_share": float(retail.loc[positive, "Description"].isna().mean()),
                "names_22776": names}
    if number == 14:
        q1, q3 = sales["Revenue"].quantile([.25, .75])
        upper = q3 + 1.5 * (q3 - q1)
        p99 = sales["Revenue"].quantile(.99)
        big = sales["Revenue"].gt(upper)
        return {"iqr_upper": float(upper), "p99": float(p99),
                "above_iqr": int(big.sum()), "above_p99": int(sales["Revenue"].gt(p99).sum()),
                "above_iqr_share": float(sales.loc[big, "Revenue"].sum() / sales["Revenue"].sum())}
    if number == 15:
        negative = retail["Quantity"].lt(0)
        return {"cancel_lines": int(cancel.sum()), "negative_lines": int(negative.sum()),
                "intersection": int((cancel & negative).sum())}
    if number == 16:
        return _country_table(sales)
    if number == 17:
        return _month_table(sales)
    if number == 18:
        grouped = sales.groupby("StockCode").agg(
            quantity=("Quantity", "sum"), revenue=("Revenue", "sum")).sort_index()
        return {"top_quantity": grouped.sort_values(
                    "quantity", ascending=False, kind="stable").head(10)["quantity"],
                "top_revenue": grouped.sort_values(
                    "revenue", ascending=False, kind="stable").head(10)["revenue"]}
    if number == 19:
        known = sales.dropna(subset=["CustomerID"])
        totals = known.groupby("CustomerID")["Revenue"].sum().sort_index()
        n1 = math.ceil(len(totals) * .01)
        n5 = math.ceil(len(totals) * .05)
        return {"customer_revenue": totals,
                "median": float(totals.median()), "p90": float(totals.quantile(.9)),
                "p99": float(totals.quantile(.99)),
                "top1_share": float(totals.nlargest(n1).sum() / totals.sum()),
                "top5_share": float(totals.nlargest(n5).sum() / totals.sum()),
                "unknown_revenue": float(sales["Revenue"].sum() - known["Revenue"].sum())}
    if number == 20:
        top = _country_table(sales).sort_values("revenue", ascending=False).head(5).index.tolist()
        part = sales.loc[sales["Country"].isin(top)].copy()
        part["month"] = part["InvoiceDate"].dt.to_period("M").astype(str)
        wide = part.pivot_table(index="month", columns="Country", values="Revenue",
                                aggfunc="sum").sort_index().reindex(columns=sorted(top))
        long = wide.reset_index().melt(id_vars="month", var_name="Country",
                                        value_name="Revenue").sort_values(
                                            ["month", "Country"]).reset_index(drop=True)
        return {"wide": wide, "long": long}
    if number == 21:
        lookup = (retail.dropna(subset=["Description"])
                  .sort_values("InvoiceDate", kind="stable")
                  .drop_duplicates("StockCode", keep="last")
                  [["StockCode", "Description"]]
                  .rename(columns={"Description": "LatestDescription"}))
        joined = sales.head(10000).drop(columns="Description").merge(
            lookup, on="StockCode", how="left", validate="many_to_one", indicator=True)
        return {"lookup": lookup.sort_values("StockCode").reset_index(drop=True),
                "joined": joined[["StockCode", "LatestDescription", "_merge"]].reset_index(drop=True)}
    if number == 22:
        left = sales.head(1000)
        right = retail.groupby("StockCode", sort=False).head(2)[["StockCode", "Description"]]
        exploded = left.merge(right, on="StockCode", how="left")
        restored = left.merge(right.drop_duplicates("StockCode"), on="StockCode", how="left",
                              validate="many_to_one")
        extra = (exploded.groupby("StockCode").size()
                 - left.groupby("StockCode").size()).sort_values(ascending=False)
        top_extra = sorted(extra[extra.eq(extra.max())].index.astype(str))[0]
        return {"exploded_rows": len(exploded), "restored_rows": len(restored),
                "largest_extra_code": top_extra,
                "exploded_revenue": float(exploded["Revenue"].sum()),
                "restored_revenue": float(restored["Revenue"].sum())}
    if number == 23:
        early = sales.loc[sales["InvoiceDate"].lt("2011-09-01")]
        late = sales.loc[sales["InvoiceDate"].ge("2011-09-01")]
        new = sorted(set(late["StockCode"].dropna()) - set(early["StockCode"].dropna()))
        return {"new_codes": new,
                "late_revenue_share": float(late.loc[late["StockCode"].isin(new), "Revenue"].sum()
                                            / late["Revenue"].sum())}
    if number == 24:
        return _daily(sales)
    if number == 25:
        countries = _country_table(sales)
        products = sales.groupby("StockCode")["Revenue"].sum()
        months = _month_table(sales).loc[:"2011-11"]
        return {"raw_rows": len(retail), "duplicates": int(retail.duplicated().sum()),
                "top_country": str(countries["revenue"].idxmax()),
                "top_product": str(products.idxmax()),
                "best_full_month": str(months["revenue"].idxmax()),
                "positive_revenue": float(sales["Revenue"].sum())}
    raise ValueError("Номер задачи должен быть от 1 до 30")


@lru_cache(maxsize=12)
def _complex_csv(name, dates=()):
    return pd.read_csv(Path(__file__).parent / "complex_data" / name,
                       parse_dates=list(dates))


def _expected_complex(number):
    if number == 26:
        events = _complex_csv("device_events.csv", ("timestamp",))
        requests = _complex_csv("device_requests.csv", ("timestamp",))
        relevant = events.loc[events["event"].isin(["error", "reset"])]
        result = []
        for request in requests.itertuples(index=False):
            prior = relevant.loc[
                relevant["device_id"].eq(request.device_id)
                & relevant["timestamp"].le(request.timestamp)
            ].sort_values(["timestamp", "event_id"])
            state = int(not prior.empty and prior.iloc[-1]["event"] == "error")
            result.append((request.request_id, state))
        return pd.DataFrame(result, columns=["request_id", "state"]).sort_values(
            "request_id").reset_index(drop=True)

    if number == 27:
        orders = _complex_csv("orders.csv", ("created_at",))
        items = _complex_csv("order_items.csv")
        returns = _complex_csv("item_returns.csv", ("returned_at",))
        cutoff = pd.Timestamp("2026-11-01")
        returned = returns.loc[returns["returned_at"].lt(cutoff)].groupby("item_id")[
            "quantity"].sum()
        work = items.copy()
        work["kept"] = work["quantity"] - work["item_id"].map(returned).fillna(0)
        work["net_revenue"] = work["kept"] * work["unit_price"] * (
            1 - work["discount_pct"] / 100)
        sums = work.groupby("order_id").agg(net_units=("kept", "sum"),
                                             net_revenue=("net_revenue", "sum"))
        result = orders[["order_id", "status"]].merge(sums, on="order_id", how="left")
        result.loc[result["status"].eq("cancelled"), ["net_units", "net_revenue"]] = 0
        return result[["order_id", "net_units", "net_revenue"]].sort_values(
            "order_id").reset_index(drop=True)

    if number == 28:
        loans = _complex_csv("loans.csv", ("borrowed_at", "due_at"))
        actions = _complex_csv("loan_events.csv", ("timestamp", "new_due_at"))
        cutoff = pd.Timestamp("2026-11-01")
        actions = actions.loc[actions["timestamp"].lt(cutoff)].sort_values(
            ["timestamp", "action_id"])
        rows = []
        for loan in loans.itertuples(index=False):
            due = loan.due_at
            returned_at = pd.NaT
            for event in actions.loc[actions["loan_id"].eq(loan.loan_id)].itertuples(index=False):
                if event.action == "return":
                    returned_at = event.timestamp
                    break
                if event.action == "renew":
                    due = event.new_due_at
            end = returned_at if pd.notna(returned_at) else cutoff - pd.Timedelta(days=1)
            overdue = max(0, (end.normalize() - due.normalize()).days)
            rows.append((loan.loan_id, due.strftime("%Y-%m-%d"), returned_at,
                         overdue, "returned" if pd.notna(returned_at) else "open"))
        return pd.DataFrame(rows, columns=["loan_id", "final_due", "returned_at",
                                           "overdue_days", "state"]).sort_values(
                                               "loan_id").reset_index(drop=True)

    if number == 29:
        opening = _complex_csv("opening_stock.csv")
        moves = _complex_csv("stock_movements.csv", ("timestamp",))
        requests = _complex_csv("stock_requests.csv", ("timestamp",))
        opening_map = opening.set_index(["warehouse", "sku"])["quantity"]
        rows = []
        for request in requests.itertuples(index=False):
            prior = moves.loc[
                moves["warehouse"].eq(request.warehouse)
                & moves["sku"].eq(request.sku)
                & moves["timestamp"].le(request.timestamp)]
            available = int(opening_map.loc[(request.warehouse, request.sku)]
                            + prior["delta"].sum())
            rows.append((request.request_id, available,
                         int(available >= request.quantity)))
        return pd.DataFrame(rows, columns=["request_id", "available", "can_fulfill"]).sort_values(
            "request_id").reset_index(drop=True)

    if number == 30:
        tickets = _complex_csv("tickets.csv", ("opened_at",))
        events = _complex_csv("ticket_events.csv", ("timestamp",))
        cutoff = pd.Timestamp("2026-11-03")
        events = events.loc[events["timestamp"].lt(cutoff)].sort_values(
            ["timestamp", "event_id"])
        limits = {"high": 60, "normal": 240, "low": 1440}
        rows = []
        for ticket in tickets.itertuples(index=False):
            history = events.loc[events["ticket_id"].eq(ticket.ticket_id)]
            replies = history.loc[history["event"].eq("agent_reply")]
            minutes = ((replies.iloc[0]["timestamp"] - ticket.opened_at).total_seconds() / 60
                       if not replies.empty else float("nan"))
            state = "open"
            for event in history.itertuples(index=False):
                if event.event == "close":
                    state = "closed"
                elif event.event == "reopen":
                    state = "open"
            rows.append((ticket.ticket_id, minutes, state,
                         int(minutes <= limits[ticket.priority])))
        return pd.DataFrame(rows, columns=["ticket_id", "first_reply_minutes",
                                           "state", "sla_met"]).sort_values(
                                               "ticket_id").reset_index(drop=True)

    raise ValueError("Номер комплексной задачи должен быть от 26 до 30")


def _compare(got, expected, path="ans"):
    if isinstance(expected, dict):
        if not isinstance(got, dict):
            return f"{path}: ожидается словарь"
        missing = [key for key in expected if key not in got]
        if missing:
            return f"{path}: отсутствуют ключи {missing}"
        for key, value in expected.items():
            problem = _compare(got[key], value, f"{path}[{key!r}]")
            if problem:
                return problem
        return None
    if isinstance(expected, pd.DataFrame):
        if not isinstance(got, pd.DataFrame):
            return f"{path}: ожидается DataFrame"
        try:
            pd.testing.assert_frame_equal(got, expected, check_dtype=False,
                check_names=False, check_index_type=False, check_column_type=False,
                check_exact=False, rtol=1e-8, atol=1e-6)
        except AssertionError as exc:
            return f"{path}: таблица отличается; {str(exc).splitlines()[0]}"
        return None
    if isinstance(expected, pd.Series):
        if not isinstance(got, pd.Series):
            return f"{path}: ожидается Series"
        try:
            pd.testing.assert_series_equal(got, expected, check_dtype=False,
                check_names=False, check_index_type=False, check_exact=False,
                rtol=1e-8, atol=1e-6)
        except AssertionError as exc:
            return f"{path}: столбец отличается; {str(exc).splitlines()[0]}"
        return None
    if isinstance(expected, list):
        if not isinstance(got, list) or len(got) != len(expected):
            return f"{path}: ожидается список из {len(expected)} элементов"
        for i, value in enumerate(expected):
            problem = _compare(got[i], value, f"{path}[{i}]")
            if problem:
                return problem
        return None
    if isinstance(expected, (float, int, np.number)) and not isinstance(expected, bool):
        if not isinstance(got, (float, int, np.number)) or not np.isclose(
                got, expected, rtol=1e-8, atol=1e-6, equal_nan=True):
            return f"{path}: получено {got!r}, ожидалось {expected!r}"
        return None
    if pd.isna(expected):
        return None if pd.isna(got) else f"{path}: ожидался пропуск"
    return None if got == expected else f"{path}: получено {got!r}, ожидалось {expected!r}"


def test_task_N(number, ans):
    """Проверить готовый ответ: test_task_N(1, ans). Возвращает True/False."""
    if not isinstance(number, int) or number not in range(1, 31):
        raise ValueError("Номер задачи должен быть целым числом от 1 до 30")
    try:
        expected = _expected(number)
        problem = _compare(ans, expected)
    except (FileNotFoundError, OSError) as exc:
        print(f"Задача {number}: {exc}")
        return False
    if problem:
        print(f"❌ Задача {number}: {problem}")
        return False
    if number <= 25:
        print(f"✅ Задача {number}: проверяемая часть верна. График и вывод проверьте отдельно.")
    else:
        print(f"✅ Задача {number}: ответ верен.")
    return True
