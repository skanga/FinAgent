# Documentation Index

## Quick Navigation

This index helps you find the right documentation for your needs.

---

## 📖 Core Documentation

### [README.md](README.md) - **START HERE**
**Purpose:** Main entry point, quick start guide, and overview
**When to read:** First time setup, basic usage
**Topics:**
- Installation instructions
- Basic configuration
- CLI examples
- Quick start guide
- Troubleshooting basics

### [ARCHITECTURE.md](ARCHITECTURE.md) - **System Design**
**Purpose:** Complete system architecture with diagrams
**When to read:** Understanding how the system works, planning modifications
**Topics:**
- System overview with ASCII diagrams
- Component architecture (Data, Analysis, LLM, Visualization layers)
- Execution flow (single ticker vs portfolio)
- Error handling (match/case patterns)
- Concurrency model (ThreadPoolExecutor, thread safety)
- Connection pooling (httpx, yfinance)
- Performance optimizations (caching, DataFrame ops, chart generation)
- Data models and design patterns
- Security considerations
- Monitoring and observability
- Extensibility points

### [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) - **Configuration Guide**
**Purpose:** Comprehensive configuration scenarios and examples
**When to read:** Setting up different environments, deployment, optimization
**Topics:**
- Basic setup (minimal vs recommended)
- LLM provider configuration (OpenAI, Claude, Gemini, Ollama)
- Performance tuning (high-performance, low-resource, rate-limited)
- Cache configuration (development, production, custom directory)
- Custom analysis parameters (benchmarks, risk-free rate, periods)
- Programmatic configuration (custom Config class, batch analysis)
- Production deployment (Docker, Kubernetes, AWS Lambda)
- Environment-specific configs (dev, staging, production)
- Configuration validation

### [RETRY_BEHAVIOR.md](RETRY_BEHAVIOR.md) - **Network Resilience**
**Purpose:** Detailed retry logic and error handling documentation
**When to read:** Network issues, API failures, customizing retry behavior
**Topics:**
- Why retry logic matters
- Retry configuration (exponential backoff, attempt limits)
- Implementation locations (fetcher.py, llm_interface.py)
- What gets retried vs what doesn't
- Error handling integration with match/case
- Retry logging (tenacity, application logs)
- Performance considerations (cache first, timeout config)
- Customization options (environment variables, decorators, callbacks)
- Testing retry behavior (unit tests, integration tests)
- Best practices and troubleshooting

---

## 🛠️ Development Documentation

### [CLAUDE.md](CLAUDE.md) - **Development Guide**
**Purpose:** Guide for developers and Claude Code users
**When to read:** Contributing to the project, understanding code patterns
**Topics:**
- Project overview
- Architecture details
- Code organization
- Development patterns
- Important implementation details
- Running the application
- Development notes

### [TESTING_QUICK_START.md](TESTING_QUICK_START.md) - **Testing Guide**
**Purpose:** How to run tests and understand test coverage
**When to read:** Running tests, adding new features, debugging
**Topics:**
- Test suite overview
- Running specific tests
- Test categories
- Writing new tests
- Test best practices

### [FEATURE_COMPARATIVE_ANALYSIS.md](FEATURE_COMPARATIVE_ANALYSIS.md) - **Feature Documentation**
**Purpose:** Detailed documentation for comparative analysis feature
**When to read:** Understanding comparative analysis, benchmark comparisons
**Topics:**
- Feature overview
- Implementation details
- Usage examples
- Design decisions

---

## 🎯 Quick Reference by Task

### I want to...

#### **Get Started**
→ Read [README.md](README.md)
→ Follow Quick Start section
→ Install dependencies and run first analysis

#### **Understand How It Works**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md)
→ Review system diagrams
→ Understand data flow

#### **Configure for Production**
→ Read [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md)
→ See Docker/Kubernetes examples
→ Review environment-specific configs

#### **Use a Different LLM Provider**
→ Read [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) > LLM Provider Configuration
→ Find your provider (OpenAI, Claude, Gemini, Ollama)
→ Copy configuration example

#### **Troubleshoot Network Errors**
→ Read [RETRY_BEHAVIOR.md](RETRY_BEHAVIOR.md)
→ Check "Troubleshooting" section
→ Enable verbose logging

#### **Optimize Performance**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md) > Performance Optimizations
→ Read [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) > Performance Tuning
→ Adjust MAX_WORKERS and cache settings

#### **Deploy to Docker**
→ Read [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) > Production Deployment > Docker
→ Copy Dockerfile
→ Follow deployment instructions

#### **Deploy to Kubernetes**
→ Read [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) > Production Deployment > Kubernetes
→ Use ConfigMap and Secret examples
→ Apply deployment YAML

#### **Deploy to AWS Lambda**
→ Read [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) > Production Deployment > AWS Lambda
→ Use lambda_handler example
→ Set environment variables

#### **Contribute Code**
→ Read [CLAUDE.md](CLAUDE.md)
→ Read [README.md](README.md) > Contributing
→ Review code patterns and conventions

#### **Run Tests**
→ Read [TESTING_QUICK_START.md](TESTING_QUICK_START.md)
→ Run test commands
→ Check test coverage

#### **Understand Error Handling**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md) > Error Handling Architecture
→ Review match/case patterns
→ Read [RETRY_BEHAVIOR.md](RETRY_BEHAVIOR.md) for network errors

#### **Understand Concurrency**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md) > Concurrency Model
→ Review thread safety measures
→ Understand worker configuration

#### **Customize Analysis**
→ Read [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) > Custom Analysis Parameters
→ Change benchmarks, risk-free rate, periods
→ Review programmatic configuration

#### **Add New Features**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md) > Extensibility Points
→ Follow patterns for indicators, providers, charts
→ Read [CLAUDE.md](CLAUDE.md) for development guide

---

## 📊 Documentation Map by Topic

### Configuration
- **Basic:** [README.md](README.md) > Configuration
- **Advanced:** [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md)
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md) > Configuration Architecture

### Performance
- **Optimization:** [ARCHITECTURE.md](ARCHITECTURE.md) > Performance Optimizations
- **Tuning:** [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) > Performance Tuning
- **Concurrency:** [ARCHITECTURE.md](ARCHITECTURE.md) > Concurrency Model
- **Connection Pooling:** [ARCHITECTURE.md](ARCHITECTURE.md) > Performance Optimizations > Connection Pooling

### Error Handling
- **Troubleshooting:** [README.md](README.md) > Troubleshooting
- **Retry Logic:** [RETRY_BEHAVIOR.md](RETRY_BEHAVIOR.md)
- **Error Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md) > Error Handling Architecture

### Deployment
- **Docker:** [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) > Production Deployment > Docker
- **Kubernetes:** [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) > Production Deployment > Kubernetes
- **AWS Lambda:** [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) > Production Deployment > AWS Lambda

### Development
- **Overview:** [CLAUDE.md](CLAUDE.md)
- **Testing:** [TESTING_QUICK_START.md](TESTING_QUICK_START.md)
- **Contributing:** [README.md](README.md) > Contributing
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)

### Features
- **Comparative Analysis:** [FEATURE_COMPARATIVE_ANALYSIS.md](FEATURE_COMPARATIVE_ANALYSIS.md)
- **All Features:** [README.md](README.md) > Features

---

## 📝 Documentation Status

| Document | Status | Last Updated | Completeness |
|----------|--------|--------------|--------------|
| README.md | ✅ Current | 2024 | 100% |
| ARCHITECTURE.md | ✅ Current | 2024 | 100% |
| CONFIG_EXAMPLES.md | ✅ Current | 2024 | 100% |
| RETRY_BEHAVIOR.md | ✅ Current | 2024 | 100% |
| CLAUDE.md | ✅ Current | 2024 | 95% |
| TESTING_QUICK_START.md | ✅ Current | 2024 | 90% |
| FEATURE_COMPARATIVE_ANALYSIS.md | ✅ Current | 2024 | 90% |

---

## 🔄 Documentation Maintenance

### Recently Consolidated (2024)
The following documents were merged into existing documentation to reduce duplication:

- ❌ **RETRY_LOGIC_IMPLEMENTATION.md** → Merged into [RETRY_BEHAVIOR.md](RETRY_BEHAVIOR.md)
- ❌ **CONNECTION_POOLING.md** → Merged into [ARCHITECTURE.md](ARCHITECTURE.md) > Performance Optimizations
- ❌ **CONCURRENT_PROCESSING.md** → Merged into [ARCHITECTURE.md](ARCHITECTURE.md) > Concurrency Model
- ❌ **ERROR_HANDLING_IMPROVEMENTS.md** → Outdated suggestions, removed (match/case now documented in ARCHITECTURE.md)

### Finding Outdated Information
If you find outdated documentation:
1. Check the main [ARCHITECTURE.md](ARCHITECTURE.md) first (most comprehensive)
2. Cross-reference with actual code implementation
3. Open an issue or submit a PR with corrections

---

## 🎓 Learning Path

### Beginner
1. [README.md](README.md) - Get familiar with basics
2. Try Quick Start examples
3. Read [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) for your LLM provider

### Intermediate
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand system design
2. [RETRY_BEHAVIOR.md](RETRY_BEHAVIOR.md) - Learn error handling
3. [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) - Advanced configuration

### Advanced
1. [CLAUDE.md](CLAUDE.md) - Development patterns
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Deep dive into all components
3. [TESTING_QUICK_START.md](TESTING_QUICK_START.md) - Test infrastructure
4. [FEATURE_COMPARATIVE_ANALYSIS.md](FEATURE_COMPARATIVE_ANALYSIS.md) - Feature design

---

## 🔗 External Resources

- **yfinance Documentation:** https://github.com/ranaroussi/yfinance
- **LangChain Documentation:** https://python.langchain.com/
- **pandas Documentation:** https://pandas.pydata.org/
- **httpx Documentation:** https://www.python-httpx.org/
- **tenacity Documentation:** https://tenacity.readthedocs.io/

---

**Last Updated:** 2024
**Maintained by:** Financial Reporting Agent Team
