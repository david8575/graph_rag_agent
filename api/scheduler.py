from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler(timezone="Asia/Seoul")

def start_scheduler():
    from api.routes.pipeline import _run_pipeline
    scheduler.add_job(
        _run_pipeline,
        "cron",
        hour=9, minute=0,
        id="daily_pipeline",
        replace_existing=True
    )
    scheduler.start()
    print("[scheduler] started - daily pipeline at 09:00 KST")

def stop_scheduler():
    scheduler.shutdown()
    print("[scheduler] stopped")
