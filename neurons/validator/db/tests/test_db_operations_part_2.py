from neurons.validator.db.tests.test_utils import TestDbOperationsBase


class TestDbOperationsPart2(TestDbOperationsBase):
    async def test_vacuum_database(self, db_operations):
        result = await db_operations.vacuum_database(500)

        assert result is None
