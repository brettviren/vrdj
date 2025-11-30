from contextlib import contextmanager

@contextmanager
def sqlite_cursor(connection):
    """
    A context manager for an sqlite3.Cursor object.
    It automatically creates a cursor upon entering the 'with' block
    and closes it upon exiting.
    """
    cursor = None
    try:
        cursor = connection.cursor()
        yield cursor
    finally:
        if cursor:
            cursor.close()
            connection.commit()
