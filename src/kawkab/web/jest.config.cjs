module.exports = {
  testEnvironment: 'jest-environment-jsdom',
  testMatch: ['**/tests/**/*.test.{js,mjs}', '**/js/__tests__/**/*.test.{js,mjs}'],
  testPathIgnorePatterns: ['e2e\\.test\\.js$'],
  moduleNameMapper: {
    '\\.(css|less|scss)$': '<rootDir>/__mocks__/styleMock.js'
  },
  transform: {},
  // Parallel worker processes crash with a StackOverflowException on this
  // combination of Node/jest-environment-jsdom versions when the full
  // suite runs (each suite passes fine alone or with --runInBand). Forcing
  // single-worker execution avoids the crash; the suite is small enough
  // (~135 tests) that this costs a fraction of a second.
  maxWorkers: 1,
};
