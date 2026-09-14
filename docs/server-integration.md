# Server integration

Install a pinned CoinCounter package release. Initialize one counter per worker at startup,
reuse it for requests, and close at shutdown. Serialize calls unless an engine documents
thread safety. Map public exceptions to HTTP responses in the server repository.

The package owns neither HTTP routes nor Docker deployment. Random guess needs explicit
count frequencies, not access to a dataset directory. Optional engine dependencies,
model loading, and remote credentials will be added with their respective engines.
